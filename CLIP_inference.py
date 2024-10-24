import torch
import numpy as np
from torch.nn.functional import cosine_similarity

class CLIPInference:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.device = config['device']
        self.model.eval()
        
        # Create data loader for collecting embeddings
        self.dataloader = torch.utils.data.DataLoader(
            dataset.data,
            batch_size=1,  # Process one at a time to maintain index correspondence
            shuffle=False
        )
        
        # Create embedding database from validation set
        self.text_embeddings = []
        self.image_embeddings = []
        
        print("Building embedding database from validation set...")
        with torch.no_grad():
            for image_embedding, text_embedding in self.dataloader:
                # These are already encoded by the dataset class
                self.image_embeddings.append(image_embedding.squeeze(0))  # Remove batch dimension
                self.text_embeddings.append(text_embedding.squeeze(0))   # Remove batch dimension
                
        # Stack all embeddings
        self.image_embeddings = torch.stack(self.image_embeddings).to(self.device)
        self.text_embeddings = torch.stack(self.text_embeddings).to(self.device)
        
        print(f"Collected {len(self.image_embeddings)} image-text pairs")
        
    def find_closest_pairs(self, query_embedding, reference_embeddings, k=5):
        """Find k closest embeddings using cosine similarity"""
        with torch.no_grad():
            # Normalize embeddings for cosine similarity
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
            reference_embeddings = reference_embeddings / reference_embeddings.norm(dim=-1, keepdim=True)
            
            similarities = torch.matmul(query_embedding.unsqueeze(0), reference_embeddings.T).squeeze(0)
            top_k_similarities, top_k_indices = torch.topk(similarities, min(k, len(reference_embeddings)))
            
        return top_k_similarities, top_k_indices
    
    def image_to_text_search(self, image_idx, k=5):
        """Find k most similar text embeddings for a given image"""
        if image_idx >= len(self.image_embeddings):
            raise ValueError(f"Image index {image_idx} out of range. Max index is {len(self.image_embeddings)-1}")
            
        query_image_embedding = self.image_embeddings[image_idx]
        
        similarities, indices = self.find_closest_pairs(
            query_image_embedding,
            self.text_embeddings,
            k=k
        )
        
        return {
            'query_idx': image_idx,
            'top_matches': [(idx.item(), sim.item()) for idx, sim in zip(indices, similarities)]
        }
    
    def text_to_image_search(self, text_idx, k=5):
        """Find k most similar images for a given text"""
        if text_idx >= len(self.text_embeddings):
            raise ValueError(f"Text index {text_idx} out of range. Max index is {len(self.text_embeddings)-1}")
            
        query_text_embedding = self.text_embeddings[text_idx]
        
        similarities, indices = self.find_closest_pairs(
            query_text_embedding,
            self.image_embeddings,
            k=k
        )
        
        return {
            'query_idx': text_idx,
            'top_matches': [(idx.item(), sim.item()) for idx, sim in zip(indices, similarities)]
        }
    
    def evaluate_retrieval_accuracy(self, num_samples=100):
        """Evaluate retrieval accuracy on a subset of validation data"""
        correct_top1 = 0
        correct_top5 = 0
        
        # Make sure we don't try to evaluate more samples than we have
        max_samples = min(num_samples, len(self.image_embeddings))
        indices = np.random.choice(len(self.image_embeddings), max_samples, replace=False)
        
        for idx in indices:
            # Image to text retrieval
            image_results = self.image_to_text_search(idx, k=5)
            top_matches = image_results['top_matches']
            
            # Check if correct text is in top-1 and top-5
            if top_matches[0][0] == idx:
                correct_top1 += 1
            if idx in [match[0] for match in top_matches]:
                correct_top5 += 1
        
        return {
            'top1_accuracy': correct_top1 / len(indices),
            'top5_accuracy': correct_top5 / len(indices),
            'num_samples': len(indices)
        }

    def get_sample_pairs(self, indices):
        """Get the actual image-text pairs for given indices"""
        pairs = []
        for idx in indices:
            pairs.append({
                'image_embedding': self.image_embeddings[idx],
                'text_embedding': self.text_embeddings[idx],
                'index': idx
            })
        return pairs

