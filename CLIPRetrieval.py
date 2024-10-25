import torch
from CLIP_model import ImageEncoder, TextEncoder

class CLIPRetrieval:
    def __init__(self, config, model, dataset):
        self.config = config
        self.model = model.to(config['device'])
        self.dataset = dataset
        self.device = config['device']
        self.model.eval()

        self.build_dictionnaries()
        
    def build_dictionnaries(self):
         # Efficiently populate embeddings, labels, and paths in one loop
        image_embeddings, text_embeddings, labels, image_paths = [], [], [], []
        
        for image, text, label, path in self.dataset:
            image_embeddings.append(image)
            text_embeddings.append(text)
            labels.append(label)
            image_paths.append(path)
        
        # Stack embeddings and move them to the specified device
        self.image_embeddings = torch.stack(image_embeddings).to(self.device)
        self.text_embeddings = torch.stack(text_embeddings).to(self.device)
        self.labels = labels
        self.image_paths = image_paths

    def find_similar(self, query, embeddings, k=5):
        with torch.no_grad():
            database_embeddings = embeddings.to(self.device)
            
            # Compute similarities with all images
            similarities = torch.matmul(query, database_embeddings.T).squeeze(0)
            
            # Get top k matches
            top_k_similarities, top_k_indices = torch.topk(similarities, k)
            
            return {
                'indices': top_k_indices.cpu().numpy(),
                'similarities': top_k_similarities.cpu().numpy(),
                'labels': [self.labels[idx] for idx in top_k_indices],
                'paths': [self.image_paths[idx] for idx in top_k_indices]
            }
    
    
    def retrieve_similar_content(self, sample, k=5):

        image_tensor, text_tensor, path, label = sample
        
        print("IMAGE-TO-TEXT RETRIEVAL")
        # query_embedding = ImageEncoder(self.config).to(self.device)       ## Encode the image
        query_embedding = image_tensor.to(self.device)
        similar_images = self.find_similar(query_embedding, self.image_embeddings, k=k)
        
        print(f"Original image label is {label}")
        print("\nTop similar items are:")
        for i, (idx, sim, path, label) in enumerate(zip(similar_images['indices'], similar_images['similarities'], similar_images['labels'], similar_images['paths'])):
            print(f"{i+1}. Image {idx} with similarity: {sim:.3f} | Label: {label} | Path: {None}")
        

        print("\nTEXT-TO-IMAGE RETRIEVAL")
        # query_embedding = TextEncoder(self.config).to(self.device)       ## Encode the text
        query_embedding = text_tensor.to(self.device)
        similar_texts = self.find_similar(query_embedding, self.text_embeddings, k=k)
        
        print(f"Original image path is {path}")
        print("Top similar items are:")
        for i, (idx, sim, path, label) in enumerate(zip(similar_texts['indices'], similar_texts['similarities'], similar_texts['labels'], similar_texts['paths'])):
            print(f"{i+1}. Image {idx} with similarity: {sim:.3f} | Label: {label} | Path: {None}")

        return { 'similar_images': similar_images, 'matching_text': similar_texts}
