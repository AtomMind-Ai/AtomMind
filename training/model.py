"""
model.py

Initializes Core and Subnet SLMs from scratch using Hugging Face GPT2Config,
with optional adapter layers for efficient fine-tuning.
"""

import torch.nn as nn
from transformers import GPT2LMHeadModel
from training.core_slm_config import get_core_slm_config
from training.study_net_config import get_study_net


class Adapter(nn.Module):
    """
    Lightweight adapter module (Houlsby-style).
    Reduces dimensionality -> nonlinearity -> projects back up.
    """
    def __init__(self, hidden_size: int, adapter_size: int = 256):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_size)
        self.nonlin = nn.ReLU()
        self.up = nn.Linear(adapter_size, hidden_size)

    def forward(self, x):
        return x + self.up(self.nonlin(self.down(x)))  # residual connection


def add_adapters_to_model(model, adapter_size: int = 256):
    """
    Inject adapters into each Transformer block of GPT2 model.
    They are attached after the feedforward layer.

    Args:
        model (GPT2LMHeadModel): Hugging Face GPT2 model.
        adapter_size (int): Bottleneck dimension for adapters.

    Returns:
        GPT2LMHeadModel with adapters.
    """
    for block in model.transformer.h:
        block.adapter = Adapter(model.config.n_embd, adapter_size)
        # Wrap original MLP forward
        orig_forward = block.mlp.forward

        def new_forward(x, orig_forward=orig_forward, adapter=block.adapter):
            x = orig_forward(x)
            return adapter(x)

        block.mlp.forward = new_forward

    return model


def init_core_slm(tokenizer=None, use_adapters=False, adapter_size=256):
    """
    Initialize Core SLM from scratch.

    Args:
        tokenizer (PreTrainedTokenizer, optional): Tokenizer with vocab and special tokens.
        use_adapters (bool): Whether to inject adapters for fine-tuning.
        adapter_size (int): Hidden size of adapter bottleneck.

    Returns:
        GPT2LMHeadModel: Core Study Helper SLM initialized with random weights.
    """
    vocab_size = len(tokenizer) if tokenizer else 50257
    config = get_core_slm_config(vocab_size=vocab_size)

    if tokenizer:
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.pad_token_id = tokenizer.pad_token_id

    model = GPT2LMHeadModel(config)
    model.init_weights()

    if use_adapters:
        model = add_adapters_to_model(model, adapter_size)

    return model


def init_study_net_slm(tokenizer=None, use_adapters=False, adapter_size=256):
    """
    Initialize Subnet SLM from scratch.

    Args:
        tokenizer (PreTrainedTokenizer, optional): Tokenizer with vocab and special tokens.
        use_adapters (bool): Whether to inject adapters for fine-tuning.
        adapter_size (int): Hidden size of adapter bottleneck.

    Returns:
        GPT2LMHeadModel: Subnet SLM initialized with random weights.
    """
    vocab_size = len(tokenizer) if tokenizer else 50257
    config = get_study_net(vocab_size=vocab_size)

    if tokenizer:
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.pad_token_id = tokenizer.pad_token_id

    model = GPT2LMHeadModel(config)
    model.init_weights()

    if use_adapters:
        model = add_adapters_to_model(model, adapter_size)

    return model
