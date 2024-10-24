import yaml
import numpy as np
import pandas as pd
import timm

import torch
from CLIP import CLIPModel
from CLIPDataset import CLIPDataset
from train import Trainer

if __name__ == "__main__":
    device = 'cuda:2' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available else 'cpu'
    
    config = yaml.safe_load(open("./config.yaml"))
    config["device"] = device
    # print(config)

    torch.manual_seed(config["seed"])

    dataset = CLIPDataset(config, None)
    model = CLIPModel(config)
    trainer = Trainer(config, model, dataset)
    trainer.run()
