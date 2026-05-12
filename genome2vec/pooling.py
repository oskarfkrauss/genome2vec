import torch
import torch.nn as nn


class GlobalAttentionPooler(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        """
        Args:
            hidden_dim: The embedding dimension of your transformer (e.g., 2560 for 2.5B model)
            num_heads: Number of attention heads. 8 is a standard default.
        """
        super().__init__()
        
        # 1. The Learnable Global Query Token
        # Shape: [Batch, SeqLen, HiddenDim] -> [1, 1, 2560]
        self.global_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # 2. Multi-Head Attention Layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # 3. LayerNorm for gradient stability
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, chunk_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            chunk_embeddings: Tensor of shape [num_chunks, hidden_dim] (e.g., [666, 2560])
        Returns:
            pooled_embedding: Tensor of shape [hidden_dim] (e.g., [2560])
        """
        # The MHA layer expects a batch dimension. 
        # We reshape from [666, 2560] to [1, 666, 2560]
        x = chunk_embeddings.unsqueeze(0)

        # The global query cross-attends to the sequence of chunks.
        # Query = The learnable token
        # Key & Value = The chunk embeddings
        attn_output, attn_weights = self.attention(
            query=self.global_query,
            key=x,
            value=x
        )

        # Remove the dummy batch and sequence dimensions: [1, 1, 2560] -> [2560]
        pooled_embedding = self.norm(attn_output.squeeze(0).squeeze(0))
        
        return pooled_embedding