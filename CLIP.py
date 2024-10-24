import torch
from torch import nn
import torch.nn.functional as F
from Encoders import ImageEncoder, TextEncoder, ProjectionHead


class CLIPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.temperature = config["temperature"]
        self.image_embedding = config["image_embedding"]
        self.text_embedding = config["text_embedding"]

        self.image_projection = ProjectionHead(config, embedding_dim=self.image_embedding)
        self.text_projection = ProjectionHead(config, embedding_dim=self.text_embedding)
        self.temperature = self.temperature

    def forward(self, images, labels):
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(images)    ## Project embeddings to 256 dimension space
        text_embeddings = self.text_projection(labels)      ## Project embeddings to 256 dimension space

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


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()