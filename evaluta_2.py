import sys
from pathlib import Path
import os
import yaml
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model import CellMem
from transforms import build_basic_transform
from datasets import SegmentationFolderDataset
from utils import get_device

# Optional FLOPs
try:
    from ptflops import get_model_complexity_info
    HAS_PTFLOPS = True
except ImportError:
    HAS_PTFLOPS = False


def compute_binary_metrics(pred, target, eps=1e-7):
    """
    pred, target: torch tensors of shape [B, 1, H, W], binary {0,1}
    """
    pred = pred.float()
    target = target.float()

    tp = (pred * target).sum(dim=(1, 2, 3))
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3))
    tn = ((1 - pred) * (1 - target)).sum(dim=(1, 2, 3))

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
        "accuracy": acc.mean().item(),
    }


def clean_binary(mask, min_area=20):
    """
    mask: uint8 [H, W], values 0 or 255
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def binary_to_instances(binary_mask_uint8):
    """
    Convert binary mask to instance mask using distance transform + watershed.
    Returns int32 instance map with labels 0,1,2,...
    """
    if binary_mask_uint8.max() == 0:
        return np.zeros_like(binary_mask_uint8, dtype=np.int32)

    dist = cv2.distanceTransform(binary_mask_uint8, cv2.DIST_L2, 5)

    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary_mask_uint8, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    color = cv2.cvtColor(binary_mask_uint8, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers)
    markers[markers < 0] = 0

    inst = np.zeros_like(markers, dtype=np.int32)
    unique_ids = [x for x in np.unique(markers) if x > 1]
    for new_id, old_id in enumerate(unique_ids, start=1):
        inst[markers == old_id] = new_id

    return inst


def gt_binary_to_instances(gt_binary_uint8):
    """
    Convert binary GT to pseudo-instance labels using connected components.
    If you later have true instance masks, replace this.
    """
    _, labels = cv2.connectedComponents(gt_binary_uint8)
    return labels.astype(np.int32)


def compute_iou(mask1, mask2, eps=1e-6):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / (union + eps)


def compute_aji(gt, pred):
    """
    gt, pred: instance masks [H, W], 0 = background
    """
    gt_ids = [i for i in np.unique(gt) if i != 0]
    pred_ids = [i for i in np.unique(pred) if i != 0]

    paired_pred = set()
    inter_sum = 0
    union_sum = 0

    for g in gt_ids:
        g_mask = (gt == g)
        best_iou = 0.0
        best_p = None

        for p in pred_ids:
            if p in paired_pred:
                continue
            p_mask = (pred == p)
            iou = compute_iou(g_mask, p_mask)
            if iou > best_iou:
                best_iou = iou
                best_p = p

        if best_p is not None and best_iou > 0:
            p_mask = (pred == best_p)
            inter_sum += np.logical_and(g_mask, p_mask).sum()
            union_sum += np.logical_or(g_mask, p_mask).sum()
            paired_pred.add(best_p)
        else:
            union_sum += g_mask.sum()

    for p in pred_ids:
        if p not in paired_pred:
            union_sum += (pred == p).sum()

    return inter_sum / (union_sum + 1e-6)


def compute_pq(gt, pred, match_iou=0.5):
    gt_ids = [i for i in np.unique(gt) if i != 0]
    pred_ids = [i for i in np.unique(pred) if i != 0]

    matched_ious = []
    used_pred = set()

    for g in gt_ids:
        g_mask = (gt == g)
        best_iou = 0.0
        best_p = None

        for p in pred_ids:
            if p in used_pred:
                continue
            p_mask = (pred == p)
            iou = compute_iou(g_mask, p_mask)
            if iou > best_iou:
                best_iou = iou
                best_p = p

        if best_p is not None and best_iou >= match_iou:
            matched_ious.append(best_iou)
            used_pred.add(best_p)

    tp = len(matched_ious)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp

    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
    sq = np.sum(matched_ious) / (tp + 1e-6) if tp > 0 else 0.0
    pq = dq * sq

    return pq, dq, sq


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_complexity(model, image_size):
    if not HAS_PTFLOPS:
        return None, None

    macs, params = get_model_complexity_info(
        model,
        (3, image_size, image_size),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )
    return macs, params


def main(
    cfg_path=ROOT /"default.yaml",
    img_dir=ROOT / "tools/data/test/images",
    mask_dir=ROOT / "tools/data/test/masks",
    proto_path=ROOT / "tools/outputs/prototypes_k256.pt",
    ckpt_path=ROOT / "tools/outputs/seg/cellmem_ep30.pt",
    threshold=0.2,
    min_area=20,
    save_instance_examples=False,
    instance_out_dir=ROOT / "tools/outputs/instance_eval_examples",
):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg["device"])

    tfm_img = build_basic_transform(cfg["image_size"])
    tfm_mask = build_basic_transform(cfg["image_size"])

    ds = SegmentationFolderDataset(str(img_dir), str(mask_dir), tfm_img, tfm_mask)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    model = CellMem(
        embed_dim=cfg["embed_dim"],
        num_prototypes=cfg["num_prototypes"],
        top_t=cfg["top_t"]
    ).to(device)

    proto = torch.load(proto_path, map_location="cpu")
    model.set_prototypes(proto.to(device))

    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    total_params, trainable_params = count_parameters(model)
    macs, ptflops_params = get_model_complexity(model, cfg["image_size"])

    totals = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "accuracy": 0.0,
        "aji": 0.0,
        "pq": 0.0,
        "dq": 0.0,
        "sq": 0.0,
    }
    n_batches = 0
    n_instances = 0

    if save_instance_examples:
        os.makedirs(instance_out_dir, exist_ok=True)

    with torch.no_grad():
        for x, y, names in tqdm(dl, desc="Evaluating"):
            x = x.to(device)
            y = y.to(device)

            logits, _ = model(x)
            prob = torch.sigmoid(logits)
            pred = (prob > threshold).float()

            batch_metrics = compute_binary_metrics(pred, y)
            for k in ["dice", "iou", "precision", "recall", "f1", "accuracy"]:
                totals[k] += batch_metrics[k]

            batch_aji = []
            batch_pq = []
            batch_dq = []
            batch_sq = []

            for b in range(x.shape[0]):
                pred_bin = (pred[b, 0].cpu().numpy() * 255).astype(np.uint8)
                gt_bin = (y[b, 0].cpu().numpy() * 255).astype(np.uint8)

                pred_bin = clean_binary(pred_bin, min_area=min_area)

                pred_inst = binary_to_instances(pred_bin)
                gt_inst = gt_binary_to_instances(gt_bin)

                aji = compute_aji(gt_inst, pred_inst)
                pq, dq, sq = compute_pq(gt_inst, pred_inst)

                batch_aji.append(aji)
                batch_pq.append(pq)
                batch_dq.append(dq)
                batch_sq.append(sq)

                if save_instance_examples:
                    pred_vis = (pred_inst.astype(np.float32) / max(pred_inst.max(), 1) * 255).astype(np.uint8)
                    gt_vis = (gt_inst.astype(np.float32) / max(gt_inst.max(), 1) * 255).astype(np.uint8)

                    cv2.imwrite(str(Path(instance_out_dir) / f"{Path(names[b]).stem}_pred_inst.png"), pred_vis)
                    cv2.imwrite(str(Path(instance_out_dir) / f"{Path(names[b]).stem}_gt_inst.png"), gt_vis)

                n_instances += 1

            totals["aji"] += float(np.mean(batch_aji)) if batch_aji else 0.0
            totals["pq"] += float(np.mean(batch_pq)) if batch_pq else 0.0
            totals["dq"] += float(np.mean(batch_dq)) if batch_dq else 0.0
            totals["sq"] += float(np.mean(batch_sq)) if batch_sq else 0.0

            n_batches += 1

    print("\n=== Segmentation Evaluation Results ===")
    for k in ["dice", "iou", "precision", "recall", "f1", "accuracy", "aji", "pq", "dq", "sq"]:
        print(f"{k}: {totals[k] / max(n_batches, 1):.4f}")

    print("\n=== Model Complexity ===")
    print(f"Total Params: {total_params:,} ({total_params / 1e6:.2f} M)")
    print(f"Trainable Params: {trainable_params:,} ({trainable_params / 1e6:.2f} M)")
    if macs is not None:
        print(f"FLOPs / MACs: {macs}")
        print(f"Params (ptflops): {ptflops_params}")
    else:
        print("FLOPs / MACs: ptflops not installed")

    print(f"\nEvaluated images: {len(ds)}")
    if save_instance_examples:
        print(f"Saved instance examples to: {instance_out_dir}")


if __name__ == "__main__":
    main()