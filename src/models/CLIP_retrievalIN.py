import os
import textwrap
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# CLIP Retrieval for ImageNet
class CLIPRetrievalIN:
    def __init__(self, config, model, dataset):
        self.config = config
        self.device = config["device"]
        self.output_dir = config["output_dir"]

        self.model = model.to(self.device)
        self.model.eval()
        self.dataset = dataset  ## Types Tensor, Tensor, string, list
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  ## Types Tensor, Tensor, tuple, list

        self.build_dictionnaries()  ## projected_image_embeddings, projected_text_embeddings, labels, image_paths
        self.compute_baseline_statistics(sample_size=1000)
        print(f"CLIP Retrieval initialized with {len(self.text_embeddings)} samples.")

    def build_dictionnaries(self):
        image_embeddings, text_embeddings, labels, images = [], [], [], []

        with torch.no_grad():
            for idx, (image, text, image_tensor, label) in enumerate(tqdm(self.dataloader)):
                # Project image embeddings to shared space
                image_embedding = image.to(self.device)  # Add batch dimension
                image_embedding = self.model.image_projection(image_embedding).squeeze(0)
                image_embedding = F.normalize(image_embedding, dim=-1)

                # Project text embeddings to shared space
                text_embedding = text.to(self.device)  # Add batch dimension
                text_embedding = self.model.text_projection(text_embedding).squeeze(0)
                text_embedding = F.normalize(text_embedding, dim=-1)

                image_embeddings.append(image_embedding)  ## image_embedding is Tensor
                text_embeddings.append(text_embedding)  ## text_embedding is Tensor
                images.append(image_tensor[0])  ## image_tensor is Tensor
                labels.append(label[0])  ## label is string

        self.image_embeddings = torch.stack(image_embeddings).to(self.device)
        self.text_embeddings = torch.stack(text_embeddings).to(self.device)
        self.images = images
        self.labels = labels

    def compute_baseline_statistics(self, sample_size=1000):
        n_samples = min(sample_size, len(self.text_embeddings))  ## Len of text_embeddings and image_embeddings = 4045

        # Compute text-to-text similarities
        text_indices = torch.randperm(len(self.text_embeddings))[:n_samples]
        text_samples = self.text_embeddings[text_indices]  ## (n_samples, 1024)
        text_similarities = torch.matmul(text_samples, text_samples.T)

        # Compute image-to-image similarities
        image_indices = torch.randperm(len(self.image_embeddings))[:n_samples]
        image_samples = self.image_embeddings[image_indices]  ## (n_samples, 1024)
        image_similarities = torch.matmul(image_samples, image_samples.T)

        # Store separate statistics for text and image
        self.text_stats = self.compute_stats(text_similarities)
        self.image_stats = self.compute_stats(image_similarities)

    def compute_stats(self, similarities):
        return {
            "mean": similarities.mean().item(),
            "std": similarities.std().item(),
            "min": similarities.min().item(),
            "max": similarities.max().item(),
            "percentiles": {
                "25": similarities.quantile(0.25).item(),
                "50": similarities.quantile(0.50).item(),
                "75": similarities.quantile(0.75).item(),
                "90": similarities.quantile(0.90).item(),
                "95": similarities.quantile(0.95).item(),
            },
        }

    def find_similar(self, query, embeddings, modality, k=5):
        with torch.no_grad():

            modality_embeddings = embeddings.to(self.device)

            # Compute similarities with all images/texts
            similarities = torch.matmul(query, modality_embeddings.T).squeeze(0)

            # Get top k matches
            top_k_similarities, top_k_indices = torch.topk(similarities, k)
            # top_k_indices = top_k_indices[0]              ## Needed for free text query
            # top_k_similarities = top_k_similarities[0]    ## Needed for free text query

            # Add evaluation for each similarity score
            normalized_scores = [self.normalize_similarity(sim.item(), modality) for sim in top_k_similarities]
            evaluations = [self.evaluate_similarity(sim.item(), modality) for sim in top_k_similarities]

            return {
                "indices": top_k_indices.cpu().numpy(),
                "similarities": top_k_similarities.cpu().numpy(),
                "normalized_scores": normalized_scores,
                "evaluations": evaluations,
                "labels": [self.labels[idx] for idx in top_k_indices],
                "images": [self.images[idx] for idx in top_k_indices],
            }

    def normalize_similarity(self, similarity, modality="text"):
        stats = self.text_stats if modality == "text" else self.image_stats
        normalized = (similarity - stats["min"]) / (stats["max"] - stats["min"])
        # print(f"Highest similarity: {stats['max']}, Lowest similarity: {stats['min']}")
        return normalized * 100

    def evaluate_similarity(self, similarity, modality="text"):
        stats = self.text_stats if modality == "text" else self.image_stats

        if similarity >= stats["percentiles"]["95"]:
            return "Excellent match (top 5%)"
        elif similarity >= stats["percentiles"]["90"]:
            return "Very good match (top 10%)"
        elif similarity >= stats["percentiles"]["75"]:
            return "Good match (top 25%)"
        elif similarity >= stats["percentiles"]["50"]:
            return "Moderate match"
        else:
            return "Weak match"

    def save_similarity_matrix(self, sample_size=10):
        # Sample a subset of embeddings for visualization
        indices = torch.randperm(len(self.image_embeddings))[:sample_size]
        image_features = self.image_embeddings[indices]  ## (batch_size, 256)
        text_features = self.text_embeddings[indices]  ## (batch_size, 256)

        # Normalize features and computes similarity
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        similarity = torch.matmul(text_features, image_features.T).cpu().numpy()

        # Create figure
        plt.figure(figsize=(20, 14))
        plt.title("Cosine Similarity between Text and Image Features", size=30, pad=20)
        matrix = plt.imshow(similarity, vmin=similarity.min(), vmax=similarity.max(), cmap="viridis")
        plt.colorbar(matrix, label="Cosine Similarity")

        # Add ticks and remove spines
        plt.yticks(range(sample_size))
        plt.xticks(range(sample_size))
        for side in ["left", "top", "right", "bottom"]:
            plt.gca().spines[side].set_visible(False)

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, f"similarity_matrix_{timestamp}.png")
        plt.savefig(filepath)
        plt.close()

    def create_retrieval_plot(self, query_label, similar_results, query_type):
        k = len(similar_results["indices"])

        # Create a single row plot for query and similar images with increased height
        fig, axes = plt.subplots(1, k + 1, figsize=(4 * (k + 1), 7))  # Increased height to 5
        fig.suptitle(f"{query_type} Retrieval Results", fontsize=16, y=0.95)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Plot query image
        query_img = query_img.permute(1, 2, 0).cpu()  ## Image shape is (3, 224, 224)
        axes[0].imshow(query_img)
        axes[0].axis("off")
        axes[0].set_title("Query Image", fontsize=12)
        wrapped_query_label = textwrap.fill(" ".join(query_label), width=50)
        axes[0].text(0.5, -0.15, wrapped_query_label, ha="center", va="top", transform=axes[0].transAxes, fontsize=10)

        # Plot retrieved images
        for i, (idx, sim, norm_score, image) in enumerate(
            zip(
                similar_results["indices"],
                similar_results["similarities"],
                similar_results["normalized_scores"],
                similar_results["images"],
            )
        ):
            retrieved_img = image.permute(1, 2, 0).cpu()
            axes[i + 1].imshow(retrieved_img)
            axes[i + 1].axis("off")

            # Set similarity score above the image
            axes[i + 1].set_title(f"Similarity: {norm_score:.2f}%", fontsize=14)

            # Wrap the label text to ensure it doesn't exceed image width
            label = int(self.labels[idx].item())
            label = self.dataset.class_descriptions[label]
            wrapped_label = textwrap.fill(" ".join(label), width=50)
            axes[i + 1].text(
                0.5, -0.15, wrapped_label, ha="center", va="top", transform=axes[i + 1].transAxes, fontsize=10
            )

        # Adjust layout to give more space for titles and labels
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95, wspace=0.1)
        filename = f"retrieval_{query_type}_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()

        return filepath

    def retrieve_similar_content(self, k=5):
        image_tensor, text_tensor, image, label = self.dataset[18]
        label = self.dataset.class_descriptions[label]

        print("\nImage-to-Image Baseline Statistics:")
        print(f"Average similarity: {self.image_stats['mean']:.3f}")
        print(f"90th percentile: {self.image_stats['percentiles']['90']:.3f}")
        print(f"95th percentile: {self.image_stats['percentiles']['95']:.3f}")

        print("\nText-to-Text Baseline Statistics:")
        print(f"Average similarity: {self.text_stats['mean']:.3f}")
        print(f"90th percentile: {self.text_stats['percentiles']['90']:.3f}")
        print(f"95th percentile: {self.text_stats['percentiles']['95']:.3f}\n")

        print("\n-----------IMAGE-TO-TEXT RETRIEVAL-----------")
        query_embedding = image_tensor.to(self.device)
        query_embedding = self.model.image_projection(query_embedding)  ## Shape [1024]
        query_embedding = F.normalize(query_embedding, dim=-1)
        similar_images = self.find_similar(query_embedding, self.image_embeddings, modality="image", k=k)

        print(f"Original image label is '{label}'")
        print("\nTop similar items are:")
        for i, (idx, sim, norm_score, eval_result, image, l) in enumerate(
            zip(
                similar_images["indices"],
                similar_images["similarities"],
                similar_images["normalized_scores"],
                similar_images["evaluations"],
                similar_images["images"],
                similar_images["labels"],
            )
        ):
            l = int(l.item())
            print(
                f"{i+1}. Image {idx} with normalized sim {norm_score:.2f}% - {eval_result}.\n   Label: {self.dataset.class_descriptions[l]}"
            )

        # Create and save image-to-image plot
        img2img_plot = self.create_retrieval_plot(label, similar_images, "Image2Text")
        print(f"Image-to-Text retrieval plot saved to: {img2img_plot}")

        print("\n-----------TEXT-TO-IMAGE RETRIEVAL-----------")
        # sample_label = "a boy jumps into the pool"                 ## Free text query
        # sample_label = [('A group of people are backpacking through a grassy field .',), ('A group of people walk in a line through a field next to a forest .',), ('A group of people walking through a grassy field .',), ('People on a nature walk with nets and backpacks with trees to the left of them .',), ('Several people in line walking through grass with nets in hand .',)]
        # encoded_captions = []
        # for caption in sample_label:
        #     encoded_caption = self.dataset.text_encoder(caption)
        #     encoded_captions.append(encoded_caption.squeeze(0))
        # encoded_captions = torch.stack(encoded_captions)

        # text_tensor = self.dataset.text_encoder(sample_label).unsqueeze(0)   ## Free text query
        query_embedding = text_tensor.to(self.device)
        query_embedding = self.model.text_projection(query_embedding)  ## Shape [1024]
        query_embedding = F.normalize(query_embedding, dim=-1)
        similar_texts = self.find_similar(query_embedding, self.text_embeddings, modality="text", k=k)

        print(f"Original image label is '{label}'")
        print("\nTop similar items are:")
        for i, (idx, sim, norm_score, eval_result, image, l) in enumerate(
            zip(
                similar_texts["indices"],
                similar_texts["similarities"],
                similar_texts["normalized_scores"],
                similar_texts["evaluations"],
                similar_texts["images"],
                similar_texts["labels"],
            )
        ):
            l = int(l.item())
            print(
                f"{i+1}. Image {idx} with normalized sim {norm_score:.2f}% - {eval_result}.\n   Label: {self.dataset.class_descriptions[l]}"
            )

        # Create and save image-to-image plot
        # sample_label = ' '.join(l[0] for l in sample_label)
        text2img_plot = self.create_retrieval_plot(label, similar_texts, "Text2Image")
        print(f"Text-to-Image retrieval plot saved to: {text2img_plot}")
