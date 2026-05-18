import os
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import yaml

# Allow imports from project root
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from datasets import ImageFolderDataset
from transforms import build_pretrain_augment
from losses import info_nce_loss
from model import StructuralEncoder
from utils import set_seed, get_device, ensure_dir


class ContrastivePairDataset(Dataset):
    """
    Wraps CellImageDataset to return two independently augmented views
    of the same image for contrastive pretraining.
    """
    def __init__(self, image_dir, image_size=224):
        self.base_dataset = ImageFolderDataset(image_dir=image_dir, transform=None)
        self.aug = build_pretrain_augment(image_size)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img_path = self.base_dataset.paths[idx]

        img = Image.open(img_path).convert("RGB")

        view1 = self.aug(img)
        view2 = self.aug(img)

        return view1, view2, img_path


def load_config(config_path="C:/Users/Sabina/Desktop/SABINAS_APPLICATION_DOCS/BK_21_Sabina/Cell_Biology/Code_files/default.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config(str(ROOT /"default.yaml"))

    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "cuda"))

    image_size = cfg.get("image_size", 224)
    batch_size = cfg.get("batch_size", 4)
    pretrain_epochs = cfg.get("pretrain_epochs", 20)
    lr = float(cfg.get("lr", 1e-4))
    embed_dim = cfg.get("embed_dim", 128)
    temperature = float(cfg.get("temperature", 0.1))

    # You can change this path to your actual pretraining images folder
    train_image_dir = r"C:/Users/Sabina/Desktop/SABINAS_APPLICATION_DOCS/BK_21_Sabina/Cell_Biology/Code_files/tools/data/train/images"

    dataset = ContrastivePairDataset(
        image_dir=train_image_dir,
        image_size=image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = StructuralEncoder(embed_dim=embed_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ensure_dir("checkpoints")

    print(f"Using device: {device}")
    print(f"Pretraining images: {len(dataset)}")
    print(f"Epochs: {pretrain_epochs}, Batch size: {batch_size}, LR: {lr}")

    best_loss = float("inf")

    for epoch in range(pretrain_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Pretrain Epoch {epoch+1}/{pretrain_epochs}")
        for x1, x2, _ in pbar:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            _, z1 = model(x1)
            _, z2 = model(x2)

            loss = info_nce_loss(z1, z2, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{running_loss / num_batches:.4f}")

        epoch_loss = running_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}: loss = {epoch_loss:.6f}")

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "checkpoints/structure_encoder.pth")
            print("Saved best structure encoder to checkpoints/structure_encoder.pth")

    print("Pretraining finished.")


if __name__ == "__main__":
    main()