from ptflops import get_model_complexity_info
from cellmem.model import CellMem
import yaml

cfg = yaml.safe_load(open("default.yaml"))

model = CellMem(
    embed_dim=cfg["embed_dim"],
    num_prototypes=cfg["num_prototypes"],
    top_t=cfg["top_t"]
)

macs, params = get_model_complexity_info(
    model,
    (3, cfg["image_size"], cfg["image_size"]),
    as_strings=True,
    print_per_layer_stat=False
)

print("FLOPs:", macs)
print("Params:", params)