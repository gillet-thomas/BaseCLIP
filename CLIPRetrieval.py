import torch

class CLIPRetrieval:
    def __init__(self, config, model, dataset):
        self.config = config
        self.model = model.to(config['device'])
        self.dataset = dataset
        self.device = config['device']
        self.model.eval()

        self.build_dictionnaries()
        self.compute_baseline_statistics()
        
    def build_dictionnaries(self):
        image_embeddings, text_embeddings, labels, image_paths = [], [], [], []
        
        for image, text, path, label in self.dataset:
            image_embeddings.append(image)
            text_embeddings.append(text)
            image_paths.append(path)
            labels.append(label)
        
        self.image_embeddings = torch.stack(image_embeddings).to(self.device)
        self.text_embeddings = torch.stack(text_embeddings).to(self.device)
        self.labels = labels
        self.image_paths = image_paths

    def find_similar(self, query, embeddings, modality, k=5):
        with torch.no_grad():
            database_embeddings = embeddings.to(self.device)
            
            # Compute similarities with all images
            similarities = torch.matmul(query, database_embeddings.T).squeeze(0)
            
            # Get top k matches
            top_k_similarities, top_k_indices = torch.topk(similarities, k)
            
            # Add evaluation for each similarity score
            normalized_scores = [self.normalize_similarity(sim.item(), modality) for sim in top_k_similarities]
            evaluations = [self.evaluate_similarity(sim.item(), modality) for sim in top_k_similarities]
            
            return {
                'indices': top_k_indices.cpu().numpy(),
                'similarities': top_k_similarities.cpu().numpy(),
                'normalized_scores': normalized_scores,
                'evaluations': evaluations,
                'labels': [self.labels[idx] for idx in top_k_indices],
                'paths': [self.image_paths[idx] for idx in top_k_indices]
            }
    
    def compute_baseline_statistics(self):
        print("Computing baseline similarity statistics...")
        
        # Sample size for efficiency
        n_samples = min(1000, len(self.text_embeddings))
        
        # Compute text-to-text similarities
        text_indices = torch.randperm(len(self.text_embeddings))[:n_samples]
        text_samples = self.text_embeddings[text_indices]
        text_similarities = torch.matmul(text_samples, text_samples.T)
        
        # Compute image-to-image similarities
        image_indices = torch.randperm(len(self.image_embeddings))[:n_samples]
        image_samples = self.image_embeddings[image_indices]
        image_similarities = torch.matmul(image_samples, image_samples.T)
        
        # Store separate statistics for text and image
        self.text_stats = self.compute_stats(text_similarities)
        self.image_stats = self.compute_stats(image_similarities)
        
    def compute_stats(self, similarities):
        return {
            'mean': similarities.mean().item(),
            'std': similarities.std().item(),
            'min': similarities.min().item(),
            'max': similarities.max().item(),
            'percentiles': {
                '25': similarities.quantile(0.25).item(),
                '50': similarities.quantile(0.50).item(),
                '75': similarities.quantile(0.75).item(),
                '90': similarities.quantile(0.90).item(),
                '95': similarities.quantile(0.95).item(),
            }
        }
    
    def normalize_similarity(self, similarity, modality='text'):
        stats = self.text_stats if modality == 'text' else self.image_stats
        normalized = (similarity - stats['min']) / (stats['max'] - stats['min'])
        return normalized * 100
    
    def evaluate_similarity(self, similarity, modality='text'):
        stats = self.text_stats if modality == 'text' else self.image_stats
        
        if similarity >= stats['percentiles']['95']:
            return "Excellent match (top 5%)"
        elif similarity >= stats['percentiles']['90']:
            return "Very good match (top 10%)"
        elif similarity >= stats['percentiles']['75']:
            return "Good match (top 25%)"
        elif similarity >= stats['percentiles']['50']:
            return "Moderate match"
        else:
            return "Weak match"
        
    def retrieve_similar_content(self, sample, k=5):
        image_tensor, text_tensor, sample_path, sample_label = sample
        
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
        similar_images = self.find_similar(query_embedding, self.image_embeddings, modality='image', k=k)
        
        print(f"Original image label is '{sample_label}'")
        print("Top similar items are:")
        for i, (idx, sim, norm_score, eval_result, path, label) in enumerate(zip(
            similar_images['indices'], 
            similar_images['similarities'], 
            similar_images['normalized_scores'],
            similar_images['evaluations'],
            similar_images['paths'], 
            similar_images['labels']
        )):
            print(f"{i+1}. Image {idx} with similarity: {sim:.3f} ({norm_score:.1f}%) - {eval_result}.\n   Label: {label}")

        print("\n-----------TEXT-TO-IMAGE RETRIEVAL-----------")
        query_embedding = text_tensor.to(self.device)
        similar_texts = self.find_similar(query_embedding, self.text_embeddings, modality='text', k=k)
        
        print(f"Original image path is '{sample_path}'")
        print("Top similar items are:")
        for i, (idx, sim, norm_score, eval_result, path, label) in enumerate(zip(
            similar_texts['indices'], 
            similar_texts['similarities'], 
            similar_texts['normalized_scores'],
            similar_texts['evaluations'],
            similar_texts['paths'], 
            similar_texts['labels']
        )):
            print(f"{i+1}. Image {idx} with similarity: {sim:.3f} ({norm_score:.1f}%) - {eval_result}.\n   Label: {label}")

        return { 'similar_images': similar_images, 'matching_text': similar_texts}