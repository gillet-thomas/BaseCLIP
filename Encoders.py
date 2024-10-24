import timm
from torch import nn

from transformers import DistilBertModel, DistilBertConfig
from transformers import DistilBertTokenizer

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