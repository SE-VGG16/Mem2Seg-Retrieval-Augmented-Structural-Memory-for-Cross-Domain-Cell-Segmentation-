import numpy as np
import torch


def kmeans_numpy(X, K=256, iters=30, seed=42):
    """
    X: [N, D] numpy float32
    returns: centers [K, D]
    """
    X = X.astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    rng = np.random.default_rng(seed)
    N, D = X.shape

    idx = rng.choice(N, size=K, replace=False) if N >= K else rng.choice(N, size=K, replace=True)
    C = X[idx].copy()

    for _ in range(iters):
        # assign
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)   # [N, K]
        a = d2.argmin(1)

        # update
        for k in range(K):
            pts = X[a == k]
            if len(pts) > 0:
                C[k] = pts.mean(0)
            else:
                C[k] = X[rng.integers(0, N)]

    return C


class PrototypeMemory(torch.nn.Module):
    def __init__(self, embed_dim=128, K=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.K = K
        self.register_buffer("prototypes", torch.zeros(K, embed_dim))

    @torch.no_grad()
    def set_prototypes(self, proto: torch.Tensor):
        if proto.shape != (self.K, self.embed_dim):
            raise ValueError(
                f"Prototype shape mismatch: expected {(self.K, self.embed_dim)}, got {tuple(proto.shape)}"
            )
        self.prototypes.copy_(proto)