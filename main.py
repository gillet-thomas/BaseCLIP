import sys

import argparse

import numpy as np
import torch
import yaml

import wandb
from src.data.DatasetFLICKR import Flickr8kDataset
from src.data.DatasetImageNet import ImageNetDataset
from src.models.CLIP_model import CLIP
from src.models.CLIP_retrieval import CLIPRetrieval
from src.models.CLIP_retrievalIN import CLIPRetrievalIN
from src.Trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train or Evaluate fMRI Model")
    parser.add_argument(
        "name", type=str, nargs="?", default=None, help="WandB run name (optional)"
    )
    parser.add_argument(
        "--inference", action="store_true", help="Run in inference mode"
    )
    parser.add_argument("--sweep", action="store_true", help="Run WandB sweep")
    parser.add_argument(
        "--cuda", type=int, default=0, help="CUDA device to use (e.g., 0 for GPU 0)"
    )
    parser.add_argument(
        "--wandb",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Enable Weights and Biases (WandB) tracking",
    )
    return parser.parse_args()

def get_device(cuda_device):
    return (
        f"cuda:{cuda_device}"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

def set_seeds(config):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

def get_datasets(config):
    # Dynamically load dataset class based on config
    if config["dataset_name"] == "FLICKR":
        dataset_train = Flickr8kDataset(
            config, mode="train", generate_data=config["generate_data"]
        )
        dataset_val = Flickr8kDataset(config, mode="val", generate_data=False)
    elif config["dataset_name"] == "IMAGENET":
        dataset_train = ImageNetDataset(
            config, mode="train", generate_data=config["generate_data"]
        )
        dataset_val = ImageNetDataset(config, mode="val", generate_data=False)

    return dataset_train, dataset_val

def get_config(args):
    config = yaml.safe_load(
        open(
            "./configs/config.yaml"
        )
    )
    config["device"] = get_device(args.cuda)
    config.update(
        {
            "wandb_enabled": args.wandb,
            "name": args.name,
            "inference": args.inference,
            "training_enabled": not args.inference
        }
    )
    return config

if __name__ == "__main__":
    args = parse_args()
    config = get_config(args)

    wandb.init(
        project="CLIP_MIMIC_CXR",
        mode="online" if config["wandb_enabled"] else "disabled",
        config=config,
        name=config["name"],
    )

    set_seeds(config)
    dataset_train, dataset_val = get_datasets(config)
    print(f"Device: {config['device']}")

    if config["training_enabled"]:
        model = CLIP(config)
        trainer = Trainer(config, model, dataset_train, dataset_val)
        trainer.run()
    else:
        print("Training is disabled. Inference mode enabled.")
        model = CLIP(config).to(config['device'])
        model.load_state_dict(torch.load("./results/model.pth", map_location=config['device'], weights_only=True))
        retrieval = CLIPRetrieval(config, model, dataset_val)

        retrieval.retrieve_similar_content()
        retrieval.save_similarity_matrix(sample_size=100)
        retrieval.free_query_retrieval()
