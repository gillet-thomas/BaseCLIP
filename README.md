<div align="center">

# 🧠 Base CLIP — Contrastive Learning for Medical & Neuroscience Data

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9.6-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 2.4-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>

</div>

---

## 📌 Introduction

This repository contains a **simple base implementation** of a **CLIP-style contrastive learning pipeline** adapted for **medical and neuroscience data**.  
It aligns embeddings from two modalities (e.g., *images and text* or *fMRI volumes and behavioral vectors*) using a **symmetric cross-entropy objective** over cosine similarity — the **CLIP loss**.

---

## 🚀 Key Features

- Encodes **two modalities** (e.g., image/text or fMRI/behavior) into a shared embedding space via lightweight **linear projection heads**.  
- Trains using a **CLIP-style contrastive loss** to maximize similarity for matched pairs and minimize it for mismatched ones.  
- Supports **training, validation, and inference**, with **Weights & Biases** logging and checkpointing to `results/`.  
- Includes **retrieval utilities** to search nearest neighbors in the learned multimodal space.  

> 🧪 This is a **baseline implementation** — two projection heads align pretrained encoders.  
> For better results, consider **fine-tuning** the final layers of the encoders (e.g., text or fMRI encoder).  
> Current encoders are simple and can be improved (e.g., by integrating the **Swift Encoder** from `fMRI2Vec`).

---

## 📁 Project Structure

```
main.py                       # Entry point for training/inference
configs/config.yaml            # Main config (datasets, encoders, heads, training params)

src/
 ├── Trainer.py                # AMP-enabled training with WandB logging
 ├── models/
 │    ├── CLIP_model.py        # Projection heads + contrastive CLIP loss
 │    ├── Encoders.py          # Image/Text encoders (timm, DistilBERT, etc.)
 │    ├── CLIP_retrieval.py    # Retrieval (Flickr)
 │    └── CLIP_retrievalIN.py  # Retrieval (ImageNet)
 └── data/
      ├── DatasetFLICKR.py     # Flickr8k pairs
      ├── DatasetImageNet.py   # ImageNet embeddings
      ├── DatasetABCDE.py      # fMRI/sMRI + behavioral data (pain scores)
      └── DatasetABCDETime.py  # Temporal variant (optional)

results/                       # Model checkpoints
wandb/                         # WandB metadata
```

---

## 💻 Getting Started

### Train

```
python main.py "run_name" --cuda 0 --wandb true
```

Process: Loads config and dataset. Builds CLIP model and projection heads. Trains with contrastive loss. Saves checkpoint to `results/model.pth`

Tip: Set `generate_data: true` for the first run  

---

### Inference / Retrieval

```
python main.py --inference --wandb false
```

Loads `results/model.pth` and runs retrieval via `CLIPRetrieval` or `CLIPRetrievalIN`.  
Outputs similarity matrices and example nearest-neighbor retrievals.

---

## ⚙️ Configuration (configs/config.yaml)

- **Global:** `output_dir`, `generate_data`, `training_enabled`, `wandb_enabled`, `seed`
- **Dataset:** `dataset_name`, dataset paths
- **Training:** `learning_rate`, `epochs`, `batch_size`, `weight_decay`, `val_interval`
- **Model:** `projection_dim`, `image_encoder` (e.g., `resnet50`, `vit_base_patch16_224`), `text_encoder`, embedding dims

Runtime flags (from `main.py`):
- `name` → WandB run name  
- `--cuda` → GPU index  
- `--wandb` → Enable/disable logging  
- `--inference` → Skip training and run retrieval  

---

## 📊 Compatible Datasets

### Flickr8k (Image–Caption Alignment)

Set in `configs/config.yaml`:
- `dataset_name: "FLICKR"`
- `dataset_flickr`: path to Flickr root (`Images/` and `captions.txt`)
- `dataset_flickr_pickle`: optional cache for precomputed embeddings

Expected layout:
```
<dataset_flickr>/
  Images/
    1000268201_693b08cb0e.jpg
    ...
  captions.txt  # CSV-like: image, caption
```

Notes:
- If `generate_data: true`, embeddings are cached for faster reuse.

### ImageNet

- Set `dataset_name: "IMAGENET"` to validate embedding and retrieval performance.  
- Use the configuration file `configs/config_imagenet.yaml` when running ImageNet experiments for pre-set parameters and paths.

### ABCDE (fMRI/MRI + Behavioral Data)

Implemented in `src/data/DatasetABCDE.py`:
- `dataset_abcde`: NIfTI volumes, e.g. `src/data/abcde/resampled/*.nii`
- `pain_scores.xlsx`: behavioral vectors (30-D per subject)
- Encoders: `fmrisEncoder` (3D CNN) and `painEncoder` (MLP)

> Note: `main.py` currently supports `FLICKR` and `IMAGENET` by default.  
> To run ABCDE, extend `get_datasets()` or add a small driver mirroring the Flickr setup.

---

## 📊 Results

The repository includes several **example experiments and visualizations** demonstrating how well the learned embeddings align across modalities.

## 🔹 Retrieval Examples

The following retrieval tasks are implemented and can be visualized from saved outputs in `results/`:

- **Image → Image** retrieval  
  Find visually or semantically similar images in the learned embedding space.

- **Text → Image** retrieval  
  Retrieve the most relevant image(s) given a text query (e.g., caption).

- **Image → Text** retrieval  
  Retrieve the caption or description that best matches a given image.

- **Text → Text** retrieval  
  Retrieve semantically similar sentences or behavioral descriptions.

All retrieval types rely on **cosine similarity** in the shared embedding space.

---

## 🔹 Similarity Matrix

During inference, the system computes a **similarity matrix** across all image–text pairs:

!!!
results/similarity_matrix.npy
!!!

This matrix encodes the pairwise cosine similarity between every element of both modalities

---

## 📈 Experimental Notes

- Works well with **Flickr8k** and **ImageNet** for baseline image–text alignment  
- Medical datasets (fMRI/MRI) included for experimentation  
- Current performance limited by **simple encoders** and **projection-only alignment**  
- Future improvements:
  - Fine-tune encoder layers  
  - Integrate **Swift Encoder** for medical modalities  
  - Explore **deeper transformer backbones** for high-dimensional embeddings  

---

## 🧭 Summary

Base CLIP is a **foundation** for contrastive multimodal learning in neuroscience and medical imaging.  
It demonstrates the feasibility of aligning **fMRI volumes**, **behavioral data**, and **images** in a shared space — providing a stepping stone toward more powerful multimodal models.
