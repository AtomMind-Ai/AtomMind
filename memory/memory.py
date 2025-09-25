"""
Multi-Network Memory System for the SLM project.
Enhancements:
- Decay in Hebbian updates
- Normalization of embeddings
- Trainable associative memory keys + buffer-based weights
- Retriever upgraded with GRU decoder + token embeddings
- Slot-based memory with usage-based eviction
- Fully batch-capable (encoder, associative, retriever, memory system)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# Encoder
class EncoderNet(nn.Module):
    """Encodes text tokens into embeddings."""

    def __init__(self, vocab_size: int, hidden_size: int, device: str):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.device = device

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, T] token IDs
        Returns:
            [B, H] normalized vectors
        """
        x = self.embedding(tokens.to(self.device))
        _, (h, _) = self.lstm(x)         # h: [1, B, H]
        v = h.squeeze(0)                 # [B, H]
        v = v / (v.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        return v

# Associative Memory
class AssociativeNet(nn.Module):
    """
    Slot-based associative memory with Hebbian updates + decay.
    Keys are trainable. Weights are non-trainable buffers.
    Includes usage counters for eviction.
    """

    def __init__(self, hidden_size: int, num_slots: int = 256, decay: float = 0.99):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.decay = decay

        # Trainable keys (slot identifiers)
        self.keys = nn.Parameter(torch.randn(num_slots, hidden_size))

        # Buffer weights (Hebbian updates, not gradient-trained)
        self.register_buffer("weights", torch.zeros(num_slots, hidden_size))
        self.register_buffer("usage", torch.zeros(num_slots))  # usage counter

    def store(self, vector: torch.Tensor):
        """Store one vector into a slot (Hebbian-inspired update)."""
        self.store_batch(vector.unsqueeze(0))

    def store_batch(self, vectors: torch.Tensor, eviction_threshold: int = 1000):
        """Batch Hebbian-inspired update with usage tracking.
        Args:
            vectors: [B, H]
        """
        with torch.no_grad():
            v = vectors / (vectors.norm(p=2, dim=-1, keepdim=True) + 1e-8)  # [B, H]

            k_norm = self.keys / (self.keys.norm(p=2, dim=-1, keepdim=True) + 1e-8)  # [S, H]
            sim = torch.matmul(v, k_norm.t())  # [B, S]
            slot_idx = sim.argmax(dim=-1)      # [B]

            for b, idx in enumerate(slot_idx.tolist()):
                # Evict least-used if this slot is overused
                if self.usage[idx] > self.eviction_threshold:
                    idx = torch.argmin(self.usage).item()
                    self.weights[idx].zero_()
                    self.usage[idx] = 0.0

                old_weight = self.weights[idx].clone()
                new_weight = self.decay * old_weight + v[b]
                self.weights[idx] = new_weight
                self.usage[idx] += 1

    def retrieve(self, queries: torch.Tensor) -> torch.Tensor:
        """Retrieve memory by weighted similarity across slots."""
        q = queries / (queries.norm(p=2, dim=-1, keepdim=True) + 1e-8)  # [B, H]
        k = self.keys / (self.keys.norm(p=2, dim=-1, keepdim=True) + 1e-8)  # [S, H]

        # Cosine similarity via dot product
        sim = torch.matmul(q, k.t())  # [B, S]
        weights = F.softmax(sim, dim=-1)  # [B, S]

        recalled = torch.matmul(weights, self.weights)  # [B, H]
        return recalled

# Retriever (Decoder)
class RetrieverNet(nn.Module):
    """Reconstructs sequences from memory vectors using a GRU decoder (batch-capable)."""

    def __init__(self, hidden_size: int, vocab_size: int, max_len: int = 20):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.fc_in = nn.Linear(hidden_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, vectors: torch.Tensor, top_k: int = 5) -> List[List[int]]:
        """
        Decode sequences of tokens from memory vectors.
        Args:
            vectors: [B, H]
            top_k: number of candidate tokens for sampling
        Returns:
            List of decoded token IDs for each batch element
        """
        batch_size = vectors.size(0)

        # project vectors to initial hidden state: [1, B, H]
        hidden = self.fc_in(vectors).unsqueeze(0).contiguous()

        # Start input: BOS embeddings (zeros here)
        inputs = torch.zeros(batch_size, 1, self.hidden_size, device=vectors.device)

        outputs: List[List[int]] = [[] for _ in range(batch_size)]

        for _ in range(self.max_len):
            out, hidden = self.gru(inputs, hidden)   # out: [B, 1, H]
            logits = self.fc_out(out.squeeze(1))     # [B, V]
            probs = F.log_softmax(logits, dim=-1)

            # Top-k sampling per batch
            top_probs, top_idx = torch.topk(probs, k=top_k, dim=-1)    # [B, k]
            sampled = torch.multinomial(top_probs.exp(), 1).squeeze(1) # [B]
            next_tokens = top_idx[torch.arange(batch_size), sampled]   # [B]

            for b in range(batch_size):
                outputs[b].append(next_tokens[b].item())

            # Next GRU input = previous hidden output
            inputs = out

        return outputs

# Memory System
class MemorySystem(nn.Module):
    """
    Multi-network memory system with Encoder -> Associative -> Retriever.
    """

    def __init__(self, vocab_size: int, hidden_size: int, device: str,
                 num_slots: int = 256, decay: float = 0.99):
        super().__init__()
        self.encoder = EncoderNet(vocab_size, hidden_size, device)
        self.associative = AssociativeNet(hidden_size, num_slots=num_slots, decay=decay)
        self.retriever = RetrieverNet(hidden_size, vocab_size)
        self.device = device

    def store(self, tokens: torch.Tensor):
        """Store one sequence of tokens."""
        encoded = self.encoder(tokens.unsqueeze(0))[0]  # [H]
        self.associative.store(encoded)
        return encoded

    def store_batch(self, tokens: torch.Tensor):
        """Store a batch of sequences."""
        encoded = self.encoder(tokens)   # [B, H]
        self.associative.store_batch(encoded)
        return encoded

    def recall(self, tokens: torch.Tensor, top_k: int = 5) -> List[List[int]]:
        """
        Recall related memory and return decoded token IDs.
        Args:
            tokens: [B, T] token IDs
        Returns:
            List of decoded sequences (per batch element)
        """
        encoded = self.encoder(tokens)                  # [B, H]
        recalled_vec = self.associative.retrieve(encoded)  # [B, H]
        return self.retriever(recalled_vec, top_k=top_k)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, List[List[int]]]:
        """Run encode → retrieve → decode (no auto-store)."""
        encoded = self.encoder(tokens)
        recalled_vec = self.associative.retrieve(encoded)
        recalled = self.retriever(recalled_vec)
        return encoded, recalled
