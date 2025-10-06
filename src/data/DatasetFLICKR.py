import os
import pickle
from collections import defaultdict

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from src.models.Encoders import ImageEncoder, TextEncoder, TextSummarizer


class Flickr8kDataset(Dataset):
    def __init__(self, config, mode="train", generate_data=False):
        self.mode = mode
        self.config = config
        self.device = config["device"]
        self.batch_size = config["batch_size"]
        self.dataset_path = config["dataset_flickr"]
        self.dataset_pickle = config["dataset_flickr_pickle"]
        self.generate_data = generate_data

        # Define data augmentations
        self.augmentations = v2.Compose(
            [
                v2.RandomResizedCrop(size=(224, 224)),  # Random cropping
                v2.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
                v2.RandomRotation(degrees=(-30, 30)),
                v2.RandomAffine(degrees=(-30, 30), translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15),
                # v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Normalize to ResNet input standards
            ]
        )

        # Initialize encoders
        self.image_encoder = ImageEncoder(config).to(self.device)  ## Used in get_data()
        self.text_encoder = TextEncoder(config).to(self.device)  ## Used in get_data()
        self.text_summarizer = TextSummarizer()
        self.image_encoder.eval()
        self.text_encoder.eval()

        # Generate pickle dataset
        if self.generate_data:
            data = self.get_data()

        # Load and split data
        with open(self.dataset_path, "rb") as file:
            data = pickle.load(file)

        # Take only 20% of data
        # data = data[:int(0.20*len(data))]

        self.train_data, self.val_data = torch.utils.data.random_split(data, [0.80, 0.20])
        self.data = self.train_data if mode == "train" else self.val_data
        print(f"Data initialized: {len(self.data)} {mode} samples")

    def get_data(self):
        # Read captions file
        captions_path = os.path.join(self.dataset_path, "captions.txt")
        df = pd.read_csv(captions_path, header=None, names=["image", "caption"])

        # Group captions by image
        image_captions = defaultdict(list)
        for _, row in df.iterrows():
            image_captions[row["image"]].append(row["caption"].strip())

        # Encode images and captions
        encoded_data_pairs = []
        images_path = os.path.join(self.dataset_path, "Images")
        for image_name, captions in tqdm(image_captions.items()):
            image_path = os.path.join(images_path, image_name)

            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue

            # Encode image and caption
            with torch.no_grad():
                image = self.load_image(image_path)
                encoded_image = self.image_encoder(image)

                # Encode all captions for this image
                combined_caption = " ".join(captions)
                combined_caption = self.text_summarizer(combined_caption)
                encoded_captions = self.text_encoder(combined_caption)  ## Tensor shape (1, 768) -> (768)

            # Store tensors on CPU to save GPU memory
            # Data is [2048], [768], image_path, [caption1, caption2, ...] of type Tensor, Tensor, str, list
            encoded_data_pairs.append(
                (encoded_image.squeeze(0).cpu(), encoded_captions.squeeze(0).cpu(), image_path, combined_caption)
            )

        with open(self.dataset_path, "rb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        return encoded_data_pairs

    def load_image(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = image.astype("float32") / 255.0
        image = torch.tensor(image).to(self.device)
        image = image.permute(2, 0, 1).float().unsqueeze(0)  # Shape: (1, 3, 224, 224) for ResNet encoder

        # if self.augmentations:
        #     image = image.squeeze(0).cpu()
        #     image = self.augmentations(image)
        #     image = image.unsqueeze(0).to(self.device)

        return image

    def __getitem__(self, idx):
        idx = idx % len(self.data)
        image, encoded_caption, image_path, combined_caption = self.data[idx]  ## Shapes (2048), (768), str, list

        return image, encoded_caption, image_path, combined_caption

    def __len__(self):
        # return self.batch_size * self.iterations_per_epoch if self.mode else len(self.data)
        return len(self.data)
