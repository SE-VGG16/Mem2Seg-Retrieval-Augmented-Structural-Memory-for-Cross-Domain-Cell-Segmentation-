import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model import CellMem
from transforms import build_basic_transform
from datasets import SegmentationFolderDataset
from losses import seg_loss
from utils import set_seed, get_device, ensure_dir


def load_yaml(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_structural_encoder(model, ckpt_path, device):
    """
    Loads a pure StructuralEncoder checkpoint into model.struct.
    Your uploaded structure_encoder.pth has keys like:
      net.0.weight, ..., proj.weight, proj.bias
    so it should load directly into model.struct.
    """
    if ckpt_path is None:
        print("No structural encoder checkpoint provided. Training from random initialization.")
        return

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Structural encoder checkpoint not found: {ckpt_path}")

    sd = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.struct.load_state_dict(sd, strict=False)

    print("Loaded structural encoder from:", ckpt_path)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)


def compute_binary_metrics_from_logits(logits, target, threshold=0.5, eps=1e-7):
    prob = torch.sigmoid(logits)
    pred = (prob > threshold).float()
    target = target.float()

    tp = (pred * target).sum(dim=(1, 2, 3))
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3))
    tn = ((1 - pred) * (1 - target)).sum(dim=(1, 2, 3))

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "accuracy": acc.mean().item(),
    }


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_accuracy = 0.0
    n_batches = 0

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)
        loss = seg_loss(logits, y)
        metrics = compute_binary_metrics_from_logits(logits, y, threshold=threshold)

        total_loss += loss.item()
        total_dice += metrics["dice"]
        total_iou += metrics["iou"]
        total_precision += metrics["precision"]
        total_recall += metrics["recall"]
        total_accuracy += metrics["accuracy"]
        n_batches += 1

    model.train()

    return {
        "loss": total_loss / max(n_batches, 1),
        "dice": total_dice / max(n_batches, 1),
        "iou": total_iou / max(n_batches, 1),
        "precision": total_precision / max(n_batches, 1),
        "recall": total_recall / max(n_batches, 1),
        "accuracy": total_accuracy / max(n_batches, 1),
    }


def main(
    cfg_path=ROOT / "default.yaml",
    train_img_dir=ROOT / "tools" / "data" / "train" / "images",
    train_mask_dir=ROOT / "tools" / "data" / "train" / "masks",
    val_img_dir=ROOT / "tools" / "data" / "test" / "images",
    val_mask_dir=ROOT / "tools" / "data" / "test" / "masks",
    proto_path=ROOT / "tools" / "outputs" / "prototypes_k256.pt",
    struct_ckpt_path=ROOT / "tools" / "checkpoints" / "structure_encoder.pth",
    out_dir=ROOT / "tools" / "outputs" / "seg",
):
    cfg = load_yaml(cfg_path)

    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])
    ensure_dir(out_dir)

    print("Using device:", device)

    tfm_img = build_basic_transform(int(cfg["image_size"]))
    tfm_mask = build_basic_transform(int(cfg["image_size"]))

    train_ds = SegmentationFolderDataset(
        str(train_img_dir), str(train_mask_dir), tfm_img, tfm_mask
    )
    val_ds = SegmentationFolderDataset(
        str(val_img_dir), str(val_mask_dir), tfm_img, tfm_mask
    )

    print("Train dataset size:", len(train_ds))
    print("Val dataset size:", len(val_ds))

    if len(train_ds) == 0:
        raise ValueError("Train dataset has 0 valid image-mask pairs.")
    if len(val_ds) == 0:
        raise ValueError("Validation dataset has 0 valid image-mask pairs.")

    train_dl = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    model = CellMem(
        embed_dim=int(cfg["embed_dim"]),
        num_prototypes=int(cfg["num_prototypes"]),
        top_t=int(cfg["top_t"]),
    ).to(device)

    # 1) load pretrained structural encoder
    load_structural_encoder(model, struct_ckpt_path, device)

    # 2) load matching prototypes
    proto_path = Path(proto_path)
    if not proto_path.exists():
        raise FileNotFoundError(f"Prototype file not found: {proto_path}")

    proto = torch.load(proto_path, map_location="cpu")
    model.set_prototypes(proto.to(device))
    print("Loaded prototypes with shape:", tuple(proto.shape))

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    epochs = int(cfg["epochs"])

    best_val_dice = -1.0

    model.train()
    for ep in range(epochs):
        pbar = tqdm(train_dl, desc=f"Epoch {ep+1}/{epochs}")
        train_loss = 0.0
        train_dice = 0.0
        train_iou = 0.0
        train_acc = 0.0
        n_batches = 0

        for x, y, _ in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits, _ = model(x)
            loss = seg_loss(logits, y)
            loss.backward()
            optimizer.step()

            metrics = compute_binary_metrics_from_logits(logits, y, threshold=0.5)

            train_loss += loss.item()
            train_dice += metrics["dice"]
            train_iou += metrics["iou"]
            train_acc += metrics["accuracy"]
            n_batches += 1

            pbar.set_postfix(
                train_loss=train_loss / max(n_batches, 1),
                train_dice=train_dice / max(n_batches, 1),
                train_acc=train_acc / max(n_batches, 1),
            )

        train_stats = {
            "loss": train_loss / max(n_batches, 1),
            "dice": train_dice / max(n_batches, 1),
            "iou": train_iou / max(n_batches, 1),
            "accuracy": train_acc / max(n_batches, 1),
        }

        val_stats = evaluate(model, val_dl, device, threshold=0.5)

        print(
            f"\nEpoch {ep+1}/{epochs} | "
            f"Train Loss: {train_stats['loss']:.4f}, Train Dice: {train_stats['dice']:.4f}, Train IoU: {train_stats['iou']:.4f}, Train Acc: {train_stats['accuracy']:.4f} | "
            f"Val Loss: {val_stats['loss']:.4f}, Val Dice: {val_stats['dice']:.4f}, Val IoU: {val_stats['iou']:.4f}, Val Acc: {val_stats['accuracy']:.4f}"
        )

        ckpt_path = Path(out_dir) / f"cellmem_ep{ep+1}.pt"
        torch.save(model.state_dict(), ckpt_path)

        if val_stats["dice"] > best_val_dice:
            best_val_dice = val_stats["dice"]
            best_path = Path(out_dir) / "cellmem_best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to: {best_path} (Val Dice={best_val_dice:.4f})")

    print("Done. Saved checkpoints to:", out_dir)


if __name__ == "__main__":
    main()