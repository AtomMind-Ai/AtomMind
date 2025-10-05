"""
study_net_config.py

Configuration for Subnet SLM (~3.5B parameters) using Hugging Face GPT2Config.
Defines a smaller transformer model to specialize in domains (math, science, etc.)
while staying aligned with the Core SLM tokenizer.
"""

from transformers import GPT2Config

def get_study_net(vocab_size=50257):
    """
    Create configuration for a Subnet SLM (~3.5B parameters).

    Args:
        vocab_size (int): Size of the tokenizer vocabulary (shared with Core SLM).

    Returns:
        GPT2Config: Hugging Face configuration object for the subnet.
    """
    return GPT2Config(
        n_layer=24,        # ~16 transformer layers
        n_embd=1792,       # hidden size (~1792 gives ~3.5B scale)
        n_head=16,         # number of attention heads
        n_inner=7618,      # feed-forward dimension
        vocab_size=vocab_size,
        bos_token_id=0,    # beginning of sequence
        eos_token_id=1,    # end of sequence
        pad_token_id=2     # padding token
    )
