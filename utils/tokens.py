import torch

def adjust_seq_len(tokens: torch.Tensor, max_len: int, pad_value: int = 0) -> torch.Tensor:
    """
    Adjust a token tensor to a fixed sequence length.
    
    Args:
        tokens (torch.Tensor): Tensor of shape (batch_size, seq_len).
        max_len (int): Maximum sequence length.
        pad_value (int, optional): Value to use for padding. Defaults to 0.
    
    Returns:
        torch.Tensor: Tensor with shape (batch_size, max_len).
    """
    tokens = tokens.long()
    seq_len = tokens.size(1)

    if seq_len > max_len:
        tokens = tokens[:, :max_len]
    elif seq_len < max_len:
        pad_len = max_len - seq_len
        padding = torch.full(
            (tokens.size(0), pad_len),
            pad_value,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        tokens = torch.cat((tokens, padding), dim=1)

    return tokens
