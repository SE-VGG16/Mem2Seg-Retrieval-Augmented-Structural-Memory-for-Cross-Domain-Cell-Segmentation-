import torch
import torch.nn as nn

from unet import UNetLite
from retrieval import retrieve_topT


class StructuralEncoder(nn.Module):
    """
    Lightweight structural encoder that outputs:
      E: [B, D, H', W']
      q: [B, D]
    """
    def __init__(self, in_ch=3, base=32, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(base * 4, embed_dim, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        f = self.net(x)
        E = self.proj(f)
        q = self.pool(E).flatten(1)
        return E, q


class RetrievalGuidedFusion(nn.Module):
    """
    Inject retrieved structural prior z into region features F_r.
    """
    def __init__(self, feat_ch=32, embed_dim=128):
        super().__init__()
        self.to_gate = nn.Sequential(
            nn.Linear(embed_dim, feat_ch),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, F_r, z):
        # F_r: [B, C, H, W], z: [B, D]
        gate = self.to_gate(z).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return F_r + self.gamma * (F_r * gate)


class CellMem(nn.Module):
    def __init__(self, embed_dim=128, num_prototypes=256, top_t=8, base=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_prototypes = num_prototypes
        self.top_t = top_t
        self.base = base

        self.struct = StructuralEncoder(in_ch=3, base=base, embed_dim=embed_dim)
        self.region = UNetLite(in_ch=3, base=base)
        self.fusion = RetrievalGuidedFusion(feat_ch=base, embed_dim=embed_dim)
        self.decoder_head = nn.Conv2d(base, 1, 1)

        self.register_buffer("prototypes", torch.zeros(num_prototypes, embed_dim))

    @torch.no_grad()
    def set_prototypes(self, proto: torch.Tensor):
        if proto.ndim != 2:
            raise ValueError(f"Prototypes must be [K, D], got shape {tuple(proto.shape)}")
        if proto.shape[1] != self.embed_dim:
            raise ValueError(
                f"Prototype embedding dim mismatch: expected {self.embed_dim}, got {proto.shape[1]}"
            )
        if proto.shape[0] != self.num_prototypes:
            raise ValueError(
                f"Prototype count mismatch: model expects {self.num_prototypes}, got {proto.shape[0]}"
            )

        self.prototypes.copy_(proto)

    def forward(self, x):
        _, q = self.struct(x)  # [B, D]
        z, idx, w = retrieve_topT(q, self.prototypes, T=self.top_t)

        F_r = self.region(x)   # [B, base, H, W]
        F_ref = self.fusion(F_r, z)
        logits = self.decoder_head(F_ref)  # [B, 1, H, W]

        return logits, {
            "z": z,
            "idx": idx,
            "w": w,
            "q": q,
        }