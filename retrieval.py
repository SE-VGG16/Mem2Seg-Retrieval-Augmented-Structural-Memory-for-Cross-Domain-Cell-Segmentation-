import torch
import torch.nn.functional as F


def retrieve_topT(
    query: torch.Tensor,
    prototypes: torch.Tensor,
    T: int = 5,
    temperature: float = 1.0,
    eps: float = 1e-8,
):
    """
    Retrieve top-T prototypes using cosine similarity.

    Args:
        query: [B, D]
        prototypes: [K, D]
        T: number of top prototypes
        temperature: softmax temperature
        eps: numerical stability

    Returns:
        retrieved: [B, D] weighted sum of top-T prototypes
        top_idx:   [B, T] indices of selected prototypes
        weights:   [B, T] softmax weights
    """
    if query.ndim != 2:
        raise ValueError(f"query must be [B, D], got {query.shape}")
    if prototypes.ndim != 2:
        raise ValueError(f"prototypes must be [K, D], got {prototypes.shape}")

    B, Dq = query.shape
    K, Dp = prototypes.shape

    if Dq != Dp:
        raise ValueError(
            f"Embedding dimension mismatch: query={Dq}, prototypes={Dp}"
        )

    T = min(T, K)

    query_n = F.normalize(query, p=2, dim=1, eps=eps)       # [B, D]
    proto_n = F.normalize(prototypes, p=2, dim=1, eps=eps)  # [K, D]

    sim = torch.matmul(query_n, proto_n.t())                # [B, K]

    top_vals, top_idx = torch.topk(sim, k=T, dim=1)         # [B, T], [B, T]
    weights = F.softmax(top_vals / temperature, dim=1)      # [B, T]

    gathered = prototypes[top_idx]                          # [B, T, D]
    retrieved = torch.sum(gathered * weights.unsqueeze(-1), dim=1)  # [B, D]

    return retrieved, top_idx, weights