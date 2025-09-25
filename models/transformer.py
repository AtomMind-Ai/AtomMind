"""Defines a reusable Transformer block for sequence processing."""

import math
import torch
from torch import nn, Tensor

class TransformerBlock(nn.Module):
    """
    TransformerBlock wraps a stack of TransformerEncoder layers.

    Args:
        hidden_size (int): Dimensionality of input embeddings.
        num_heads (int): Number of attention heads per layer.
        num_layers (int): Number of TransformerEncoder layers.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_first: bool = True,
        use_positional_encoding: bool = False,
        max_len: int = 5000,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.register_buffer(
                "positional_encoding",
                self._build_positional_encoding(hidden_size, max_len),
                persistent=False,
            )

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through the Transformer block.

        Args:
            x (Tensor): Input tensor of shape [batch, seq_len, hidden_size].
        
        Returns:
            Tensor: Output tensor of same shape as input.
        """
        if self.use_positional_encoding:
            x = x + self.positional_encoding[:, : x.size(1)].to(x.device)

        return self.transformer(
            x, mask=attn_mask, src_key_padding_mask=src_key_padding_mask
        )

    @staticmethod
    def _build_positional_encoding(hidden_size: int, max_len: int = 5000) -> Tensor:
        """
        Generate sinusoidal positional encodings for a sequence.
    
        Positional encodings provide information about the position of tokens 
        in a sequence, allowing transformer models to incorporate the order of 
        elements without using recurrent networks.
    
        The encoding is computed using sine and cosine functions of different 
        frequencies:
    
            PE(pos, 2i)   = sin(pos / (10000^(2i/hidden_size)))
            PE(pos, 2i+1) = cos(pos / (10000^(2i/hidden_size)))
    
        Args:
            hidden_size (int): The dimensionality of the embeddings.
            max_len (int, optional): Maximum sequence length for which to compute 
                positional encodings. Default is 5000.
    
        Returns:
            Tensor: A tensor of shape [1, max_len, hidden_size] containing the 
                positional encodings, ready to be added to input embeddings.
        """
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, hidden_size]
