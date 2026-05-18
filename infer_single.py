import yaml
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from pathlib import Path
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]   # project root

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from model import CellMem
from transforms import build_basic_transform
from utils import get_device

def main(cfg_path="C:/Users\Sabina\Desktop\SABINAS_APPLICATION_DOCS\BK_21_Sabina\Cell_Biology\Code_files/default.yaml",
         img_path="3.png",
         proto_path="outputs/prototypes_k256.pt",
         ckpt_path=r"C:/Users/Sabina/Desktop/SABINAS_APPLICATION_DOCS/BK_21_Sabina/Cell_Biology/Code_files/tools/outputs/seg/cellmem_ep30.pt",
         out_dir="outputs/infer"):
    cfg = yaml.safe_load(open(cfg_path, "r"))
    device = get_device(cfg["device"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    tfm = build_basic_transform(cfg["image_size"])
    img_pil = Image.open(img_path).convert("RGB")
    x = tfm(img_pil).unsqueeze(0).to(device)

    model = CellMem(embed_dim=cfg["embed_dim"],
                    num_prototypes=cfg["num_prototypes"],
                    top_t=cfg["top_t"]).to(device).eval()

    proto = torch.load(proto_path, map_location="cpu").to(device)
    model.set_prototypes(proto)

    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd, strict=False)

    with torch.no_grad():
        logits, info = model(x)
        prob = torch.sigmoid(logits)[0,0].cpu().numpy()

    mask = (prob > 0.5).astype(np.uint8) * 255

    # Resize mask back to original image size for overlay
    orig = cv2.imread(img_path)
    mask_up = cv2.resize(mask, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay = orig.copy()
    contours, _ = cv2.findContours(mask_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0,255,255), 2)

    cv2.imwrite(str(Path(out_dir)/"mask.png"), mask_up)
    cv2.imwrite(str(Path(out_dir)/"overlay.png"), overlay)

    print("Saved:", (Path(out_dir)/"mask.png").resolve())
    print("Saved:", (Path(out_dir)/"overlay.png").resolve())
    print("Retrieved prototype indices:", info["idx"][0].cpu().tolist())

if __name__ == "__main__":
    main()