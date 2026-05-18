import yaml
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from model import StructuralEncoder
from memory import kmeans_numpy
from transforms import build_basic_transform
from datasets import ImageFolderDataset
from utils import set_seed, get_device

def main(cfg_path="C:/Users\Sabina\Desktop\SABINAS_APPLICATION_DOCS\BK_21_Sabina\Cell_Biology\Code_files/default.yaml", image_dir="data/train/images", out_path="outputs/prototypes_k256.pt"):
    cfg = yaml.safe_load(open(cfg_path, "r"))
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    tfm = build_basic_transform(cfg["image_size"])
    ds = ImageFolderDataset(image_dir, tfm)

    enc = StructuralEncoder(embed_dim=cfg["embed_dim"]).to(device).eval()

    all_q = []
    with torch.no_grad():
        for x, _ in tqdm(ds, desc="Extract embeddings"):
            x = x.unsqueeze(0).to(device)
            _, q = enc(x)
            all_q.append(q.squeeze(0).cpu().numpy().astype(np.float32))

    X = np.stack(all_q, axis=0)  # [N,D]
    centers = kmeans_numpy(X, K=cfg["num_prototypes"], iters=30, seed=cfg["seed"])
    proto = torch.from_numpy(centers).float()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(proto, out_path)
    print("Saved prototypes:", out_path, proto.shape)

if __name__ == "__main__":
    main()