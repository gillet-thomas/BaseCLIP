import sys
import yaml
import wandb
import torch

from src.Trainer import Trainer
from src.CLIP_model import CLIP
from src.CLIP_retrieval import CLIPRetrieval
from src.data.MIMIC import MIMICDataset

if __name__ == "__main__":
    device = 'cuda:2' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available else 'cpu'
    config = yaml.safe_load(open("./configs/config.yaml"))
    config["device"] = device

    torch.manual_seed(config["seed"])

    # Initialize wandb
    args = sys.argv[1:]
    name = args[0] if len(args) > 0 else None
    wandb_mode = 'online' if config["wandb_enabled"] == 1 else 'disabled'
    wandb.init(project="CLIP_MIMIC_CXR", mode=wandb_mode, config=config, name=name)

    if config['training_enabled']:
        dataset = MIMICDataset(config)
        model = CLIP(config)
        trainer = Trainer(config, model, dataset)
        trainer.run()
    else:
        print("Training is disabled. Inference mode enabled.")
        dataset = MIMICDataset(config, mode="val")
        model = CLIP(config).to(device)
        model.load_state_dict(torch.load('./results/CLIP_model.pth', map_location=device, weights_only=True))
        model.to(device)
        retrieval = CLIPRetrieval(config, model, dataset)
        results = retrieval.retrieve_similar_content(dataset[0], k=5)