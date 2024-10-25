import sys
import yaml
import wandb
import torch

from CLIP_model import CLIP
from CLIP_dataset import MIMIC
from Trainer import Trainer
from CLIPRetrieval import CLIPRetrieval

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
        model.load_state_dict(torch.load('./CLIP_model.pth', map_location=device, weights_only=True))
        model.to(device)
        # inference = ImprovedCLIPInference(config, model, dataset)
        # inference.run_inference(num_samples=1)

        # # Assuming you have the trained CLIP model, dataset, and config loaded:
        # inference = CLIPInferenceLatentSpace(model, dataset, config)

        # # Retrieve label for an image
        # image_idx = 42  # Example image index
        # label_results = inference.retrieve_label_from_image(image_idx)
        # print(f"Label retrieval for image {image_idx}: {label_results}")

        # # Retrieve image for a label
        # text_idx = 42  # Example text index
        # image_results = inference.retrieve_image_from_label(text_idx)
        # print(f"Image retrieval for label {text_idx}: {image_results}")

        # # Evaluate accuracy
        # accuracy_results = inference.evaluate_retrieval_accuracy(num_samples=100)
        # print(f"Retrieval accuracy: {accuracy_results}")

        retrieval = CLIPRetrieval(config, model, dataset)
        results = retrieval.retrieve_similar_content(dataset[0], k=5)