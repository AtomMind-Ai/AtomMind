"""
core_slm_config.py

Configuration for Core SLM (~2.5B parameters) from scratch using Hugging Face GPT2Config.
Defines a large-scale transformer suitable for academic study-helper tasks.
"""

from transformers import GPT2Config

def get_core_slm_config(vocab_size=50257):
    """
    Create configuration for the Core Study Helper SLM (~2.5B parameters).

    Args:
        vocab_size (int): Size of the tokenizer vocabulary.

    Returns:
        GPT2Config: Hugging Face configuration object.
    """
    return GPT2Config(
        n_layer=18,        # Increase to 18 layers (adding more depth)
        n_embd=1536,       # Hidden size (~1536 gives ~2.5B scale)
        n_head=16,         # Number of attention heads (can remain the same)
        n_inner=6144,      # Feed-forward dimension
        vocab_size=vocab_size,
        bos_token_id=0,    # Beginning of sequence token ID
        eos_token_id=1,    # End of sequence token ID
        pad_token_id=2     # Padding token ID
    )