"""
Defines the ChatExpertNetwork (CEN)
A transformer-based expert network for processing chat sequences.
"""

from torch import nn, Tensor
from models.transformer import TransformerBlock

class ChatExpertNetwork(nn.Module):
    """
    ChatExpertNetwork applies a stack of 
    TransformerEncoder layers to input chat embeddings.
    """

    def __init__(self, hidden_size: int, num_layers: int, num_heads: int) -> None:
        super().__init__()
        self.transformer = TransformerBlock(hidden_size, num_heads, num_layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the transformer.

        Args:
            x (Tensor): Input tensor of shape [batch, seq_len, hidden_size].
        Returns:
            Tensor: Output tensor of shape [batch, seq_len, hidden_size].
        """
        # Transformer expects [seq_len, batch, hidden_size]
        return self.transformer(x)
