# contains tools cross attention options 1, 2 and 3

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_similarity_matrix(embeddings_a: torch.Tensor, 
                              embeddings_b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between all chunks of A and B.
    
    Args:
        embeddings_a: [n_chunks_a, hidden_dim]
        embeddings_b: [n_chunks_b, hidden_dim]
    Returns:
        similarity_matrix: [n_chunks_a, n_chunks_b]
    """
    a_norm = F.normalize(embeddings_a, p=2, dim=1)
    b_norm = F.normalize(embeddings_b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.T)


def filter_divergent_chunks(similarity_matrix: torch.Tensor,
                             low_threshold: float = 0.3,
                             high_threshold: float = 0.99):
    """
    Find chunk pairs that are in the 'divergent' middle range.
    Discard identical (> high_threshold) and unrelated (< low_threshold).
    
    Returns indices of divergent chunk pairs in A and their best match in B.
    """
    # For each chunk in A, find its best matching chunk in B
    best_match_scores, best_match_indices = similarity_matrix.max(dim=1)
    
    divergent_mask = (
        (best_match_scores < high_threshold) & 
        (best_match_scores > low_threshold)
    )
    
    divergent_chunks_a = divergent_mask.nonzero(as_tuple=True)[0]
    divergent_chunks_b = best_match_indices[divergent_mask]
    
    return divergent_chunks_a, divergent_chunks_b, best_match_scores

# ---------------------------------------------------------------------------
# Option 1 — Cross attention over ALL CLS tokens
# ---------------------------------------------------------------------------
 
class Option1CrossAttention(nn.Module):
    """
    Cross attention directly over all CLS tokens from isolate A and B.
 
    A attends to B across all chunk pairs simultaneously.

 
    Args:
        hidden_dim: embedding dimension (2560 for NT 2.5B, 768 for DNABERT)
        num_heads:  number of attention heads
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
 
    def forward(self,
                emb_a: torch.Tensor,   # [n_chunks_a, hidden_dim]
                emb_b: torch.Tensor,   # [n_chunks_b, hidden_dim]
                ) -> tuple[torch.Tensor, torch.Tensor]:
        # add batch dimension: [1, n_chunks, hidden_dim]
        a = emb_a.unsqueeze(0)
        b = emb_b.unsqueeze(0)
 
        # A queries B
        attn_out, attn_weights = self.cross_attn(query=a, key=b, value=b)
 
        # pool across chunk dimension → [1, hidden_dim]
        pooled = self.norm(attn_out.mean(dim=1))
 
        # project to scalar distance
        distance = self.mlp(pooled).squeeze()
        return distance, attn_weights
 
 
# ---------------------------------------------------------------------------
# Option 2 — Cosine filter first, then cross attention on divergent chunks
# ---------------------------------------------------------------------------
 
class Option2FilteredCrossAttention(nn.Module):
    """
    Cross attention only on the divergent chunk pairs identified by
    cosine filtering. Cosine filtering is done externally using
    cosine_similarity_matrix + filter_divergent_chunks before calling
    this module.
 
    Cleaner signal than Option 1 because identical chunks are already
    removed before the attention mechanism sees them.
 
    Args:
        hidden_dim: embedding dimension
        num_heads:  number of attention heads
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
 
    def forward(self,
                div_emb_a: torch.Tensor,   # [n_divergent, hidden_dim]
                div_emb_b: torch.Tensor,   # [n_divergent, hidden_dim]
                ) -> tuple[torch.Tensor, torch.Tensor]:
        a = div_emb_a.unsqueeze(0)
        b = div_emb_b.unsqueeze(0)
 
        attn_out, attn_weights = self.cross_attn(query=a, key=b, value=b)
        pooled = self.norm(attn_out.mean(dim=1))
        distance = self.mlp(pooled).squeeze()
        return distance, attn_weights
 
 
# ---------------------------------------------------------------------------
# Option 3 — Cosine filter, then token-level cross attention
# ---------------------------------------------------------------------------
 
class Option3TokenLevelCrossAttention(nn.Module):
    """
    Cross attention at full token resolution within divergent chunks.
 
    Divergent chunks are identified by cosine filtering (same as Option 2),
    then re-embedded at the token level externally, and passed here.
 
    Most granular option — operates at nucleotide level within the regions
    that actually differ. Most likely to detect very low SNP distances (5 SNPs).
 
    Args:
        hidden_dim: embedding dimension
        num_heads:  number of attention heads
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
 
    def forward(self,
                tokens_a: torch.Tensor,   # [seq_len_a, hidden_dim] full token embeddings
                tokens_b: torch.Tensor,   # [seq_len_b, hidden_dim]
                ) -> tuple[torch.Tensor, torch.Tensor]:
        a = tokens_a.unsqueeze(0)
        b = tokens_b.unsqueeze(0)
 
        attn_out, attn_weights = self.cross_attn(query=a, key=b, value=b)
        pooled = self.norm(attn_out.mean(dim=1))
        distance = self.mlp(pooled).squeeze()
        return distance, attn_weights