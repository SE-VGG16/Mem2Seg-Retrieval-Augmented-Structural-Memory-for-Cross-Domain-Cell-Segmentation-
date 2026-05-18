import os
import sys
from pathlib import Path
import json
import math
import yaml
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageDraw
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from cellmem.model import StructuralEncoder
from cellmem.transforms import build_basic_transform
from cellmem.utils import get_device


def load_image_paths(image_dir):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(exts)]
    paths.sort()
    return paths


def encode_patches(model, paths, transform, device):
    feats = []
    valid_paths = []
    model.eval()
    with torch.no_grad():
        for p in tqdm(paths, desc="Encoding patches"):
            img = Image.open(p).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            _, q = model(x)   # q: [1, D]
            feats.append(q.squeeze(0).cpu())
            valid_paths.append(p)
    feats = torch.stack(feats, dim=0)  # [N, D]
    return feats, valid_paths


def assign_to_prototypes(feats, prototypes):
    feats = F.normalize(feats, dim=1)
    prototypes = F.normalize(prototypes, dim=1)
    sim = feats @ prototypes.t()              # [N, K]
    best_sim, best_idx = sim.max(dim=1)       # [N], [N]
    return best_idx, best_sim, sim


def make_grid(items, out_path, thumb_size=96, cols=8, pad=8, bg=(245, 245, 245)):
    """
    items: list of tuples (image_path, label_text)
    """
    if len(items) == 0:
        return

    rows = math.ceil(len(items) / cols)
    w = cols * thumb_size + (cols + 1) * pad
    h = rows * (thumb_size + 22) + (rows + 1) * pad

    canvas = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(canvas)

    for i, (img_path, label) in enumerate(items):
        r = i // cols
        c = i % cols
        x = pad + c * (thumb_size + pad)
        y = pad + r * (thumb_size + 22 + pad)

        img = Image.open(img_path).convert("RGB")
        img = ImageOps.fit(img, (thumb_size, thumb_size))
        canvas.paste(img, (x, y))
        draw.text((x, y + thumb_size + 2), label, fill=(60, 60, 60))

    canvas.save(out_path)
    print("Saved:", out_path)


def main():
    cfg_path = ROOT / "configs" / "default.yaml"
    patch_dir = ROOT / "tools/data/prototype_patches"
    encoder_ckpt = ROOT / "tools" / "checkpoints" / "structure_encoder.pth"
    proto_path = ROOT / "tools/outputs/prototypes_k64.pt"
    out_dir = ROOT / "tools/outputs/prototype_viz"

    os.makedirs(out_dir, exist_ok=True)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg["device"])
    transform = build_basic_transform(cfg["image_size"])

    model = StructuralEncoder(embed_dim=cfg["embed_dim"]).to(device)
    sd = torch.load(encoder_ckpt, map_location=device)
    model.load_state_dict(sd, strict=False)

    prototypes = torch.load(proto_path, map_location="cpu").float()

    paths = load_image_paths(str(patch_dir))
    feats, paths = encode_patches(model, paths, transform, device)
    best_idx, best_sim, sim = assign_to_prototypes(feats, prototypes)

    # Save top examples per prototype
    K = prototypes.shape[0]
    summary = {}

    for k in range(K):
        inds = torch.where(best_idx == k)[0]
        if len(inds) == 0:
            continue
        inds = inds[torch.argsort(best_sim[inds], descending=True)]
        top_inds = inds[:8]  # top 8 examples for this prototype

        items = []
        summary[k] = []
        for idx in top_inds.tolist():
            label = f"s={best_sim[idx].item():.2f}"
            items.append((paths[idx], label))
            summary[k].append({"path": paths[idx], "score": float(best_sim[idx].item())})

        make_grid(items, os.path.join(out_dir, f"prototype_{k:02d}.png"), cols=4)

    with open(os.path.join(out_dir, "prototype_examples.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()