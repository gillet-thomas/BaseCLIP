import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig

class CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.temperature = config["temperature"]
        self.image_embedding = config["image_embedding"]
        self.text_embedding = config["text_embedding"]

        self.image_projection = ProjectionHead(config, embedding_dim=self.image_embedding)
        self.text_projection = ProjectionHead(config, embedding_dim=self.text_embedding)
        self.temperature = self.temperature

    def forward(self, sources, targets):
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(sources)    ## Project embeddings to 256 dimension space, shape: (batch_size, 256)
        text_embeddings = self.text_projection(targets)      ## Project embeddings to 256 dimension space, shape: (batch_size, 256)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

# Cross Entropy Loss implementation from scratch
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    
class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(self, config):
        super().__init__()

        self.model_name = config["model_name"]
        self.pretrained = config["pretrained"]
        self.trainable = config["trainable"]

        self.model = timm.create_model(self.model_name, self.pretrained, num_classes=0, global_pool="avg")
        for p in self.model.parameters():
            p.requires_grad = self.trainable

    def forward(self, x):
        # # Get embedding and normalize
        # image_emb = self.model.image_projection(image_tensor)
        # image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        return self.model(x)
    
class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = config["device"]
        self.model_name = config["text_encoder_model"]
        self.pretrained = config["pretrained"]
        self.trainable = config["trainable"]
        self.max_length = config.get('max_length', 512)
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        
        if self.pretrained:
            self.model = DistilBertModel.from_pretrained(self.model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        self.model.to(self.device)

        for p in self.model.parameters():
            p.requires_grad = self.trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0 ## Index 0 is CLS token represented by value 101

    def forward(self, text):
        tokenized_text = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        # Move to device
        input_ids = tokenized_text['input_ids'].to(self.device)
        attention_mask = tokenized_text['attention_mask'].to(self.device)
        
        # Get the model output
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]  ## Output is shape (batch_size, hidden_size)
    
    def decode(self, encoded_ids):
        # Remove any padding tokens
        if isinstance(encoded_ids, torch.Tensor):
            encoded_ids = encoded_ids.cpu().numpy()
            
        # Remove special tokens and decode
        decoded_text = self.tokenizer.decode(encoded_ids, 
                                           skip_special_tokens=True,
                                           clean_up_tokenization_spaces=True)
        return decoded_text
    
class ProjectionHead(nn.Module):
    def __init__(self, config, embedding_dim):
        super().__init__()
        
        # Embedding dim is 2048 for image and 768 for text, projection_dim is 256
        self.projection_dim = config["projection_dim"]
        self.dropout = config["dropout"]

        self.projection = nn.Linear(embedding_dim, self.projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(self.projection_dim, self.projection_dim)
        self.dropout = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected       ## Skip connection
        x = self.layer_norm(x)
        return x