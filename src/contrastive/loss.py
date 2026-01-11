# src/contrastive/loss.py
import torch
import torch.nn.functional as F

def info_nce_loss(anchor, positive, negatives, temperature=0.1):
    """
    Calculates the InfoNCE loss for contrastive learning.
    Args:
        anchor: Embeddings of the anchor trajectories [B, E]
        positive: Embeddings of the positive trajectories [B, E]
        negatives: Embeddings of the negative trajectories [B, N, E]
    """
    # Calculate similarity between anchor and positive
    # Result shape: [B, 1]
    pos_sim = F.cosine_similarity(anchor, positive, dim=-1).unsqueeze(-1)

    # Calculate similarity between anchor and all negatives
    # Result shape: [B, N]
    neg_sim = F.cosine_similarity(anchor.unsqueeze(1), negatives, dim=-1)

    # Combine them. The positive similarity should be maximized.
    # Result shape: [B, 1 + N]
    logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature
    
    # The 'correct' label is always the first one (the positive pair)
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
    
    return F.cross_entropy(logits, labels)