import sys
import yaml
import wandb
import torch
import numpy as np
from torch.nn.functional import cosine_similarity

from CLIP_model import CLIP
from CLIP_dataset import MIMIC
from train import Trainer


class CLIPInference:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.device = config['device']
        self.model.eval()

    def get_projection_embeddings(self, image_embedding, text_embedding):
        """Project embeddings through CLIP model's projection heads"""
        with torch.no_grad():
            # Project through the model's projection heads
            projected_image = self.model.image_projection(image_embedding)
            projected_text = self.model.text_projection(text_embedding)
            
            # Normalize the embeddings
            projected_image = projected_image / projected_image.norm(dim=-1, keepdim=True)
            projected_text = projected_text / projected_text.norm(dim=-1, keepdim=True)
            
            return projected_image, projected_text

    def get_prediction(self, idx):
        """Get prediction for a single sample"""
        image_embedding, label_embedding, image_path, true_label = self.dataset[idx]
        image_embedding = image_embedding.unsqueeze(0).to(self.device)
        label_embedding = label_embedding.unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Project embeddings to same space
            proj_image, proj_label = self.get_projection_embeddings(image_embedding, label_embedding)
            
            # Calculate similarity between projected embeddings
            sim_score = cosine_similarity(proj_image, proj_label, dim=1).item()
            
            # Get comparison with nearby samples
            start_idx = max(0, idx - 2)
            end_idx = min(len(self.dataset), idx + 3)
            comparison_scores = []
            
            # Compare with nearby samples
            for i in range(start_idx, end_idx):
                comp_image, comp_label, path, label = self.dataset[i]
                comp_image = comp_image.unsqueeze(0).to(self.device)
                comp_label = comp_label.unsqueeze(0).to(self.device)
                
                # Project comparison embeddings
                proj_comp_image, proj_comp_label = self.get_projection_embeddings(comp_image, comp_label)
                
                # Calculate similarities
                image_to_label = cosine_similarity(proj_image, proj_comp_label, dim=1).item()
                comparison_scores.append((i, image_to_label, label))
            
            # Sort by similarity score
            comparison_scores.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'true_label': true_label,
                'true_label_similarity': sim_score,
                'nearby_comparisons': comparison_scores
            }

def run_simple_inference_demo(model, dataset, config, num_samples=5):
    print(f"\nRunning inference on {num_samples} random samples...")
    inference = CLIPInference(model, dataset, config)

    total_samples = len(dataset)
    indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)
    print(f"Selected indices: {indices}")
    
    for idx in indices:
        results = inference.get_prediction(idx)
        
        print(f"\nSample {idx}:")
        print("True label is: ", results['true_label'])
        print(f"Similarity with true label: {results['true_label_similarity']:.3f}")
        print("Similarities with nearby samples:")
        for comp_idx, similarity, label in results['nearby_comparisons']:
            print(f"  Index {comp_idx}: {similarity:.3f} (Label: {label})")
            if comp_idx == idx:
                print("  ✓ This is the correct match!")

if __name__ == "__main__":
    device = 'cuda:2' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available else 'cpu'
    config = yaml.safe_load(open("./config.yaml"))
    config["device"] = device

    torch.manual_seed(config["seed"])

    # Initialize wandb
    args = sys.argv[1:]
    name = args[0] if len(args) > 0 else None
    wandb_mode = 'online' if config["wandb_enabled"] == 1 else 'disabled'
    wandb.init(project="CLIP_MIMIC_CXR", mode=wandb_mode, config=config, name=name)

    if config['training_enabled']:
        model = CLIP(config)
        dataset = MIMIC(config)
        trainer = Trainer(config, model, dataset)
        trainer.run()
    else:
        print("Training is disabled. Inference mode enabled.")
        dataset = MIMIC(config, mode="val")
        model = CLIP(config).to(device)
        model.load_state_dict(torch.load('./CLIP_model.pth', map_location=device))
        model.to(device)

        run_simple_inference_demo(model, dataset, config, num_samples=5)