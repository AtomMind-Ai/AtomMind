"""
memory_system.py

Implements a neural memory system that connects Core and Study Net.
Acts like a brain-like associative memory: store, retrieve, integrate.
All retrieval + integration logic is contained in this file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemorySystem(nn.Module):
    def __init__(self, embed_dim=768, memory_size=2048, device="cuda"):
        super().__init__()
        self.device = device

        # Memory slots (key-value storage)
        self.embed_dim = embed_dim
        self.memory_size = memory_size
        self.memory_keys = torch.zeros((memory_size, embed_dim), device=device)
        self.memory_values = torch.zeros((memory_size, embed_dim), device=device)
        self.ptr = 0  # pointer for cyclic buffer

        # Fusion layer (Core + Study + Memory → unified representation)
        self.fusion = nn.Linear(embed_dim * 3, embed_dim).to(device)

    # -------------------------
    # Memory Storage
    # -------------------------
    def store(self, key, value):
        """
        Store key-value pair into memory (cyclic buffer).
        key: Tensor [embed_dim]
        value: Tensor [embed_dim]
        """
        idx = self.ptr % self.memory_size
        self.memory_keys[idx] = key.detach()
        self.memory_values[idx] = value.detach()
        self.ptr += 1

    # -------------------------
    # Memory Retrieval
    # -------------------------
    def retrieve(self, query, top_k=5):
        """
        Retrieve top-k most relevant memory values given a query.
        query: Tensor [embed_dim]
        Returns: Tensor [top_k, embed_dim]
        """
        similarities = F.cosine_similarity(
            query.unsqueeze(0), self.memory_keys, dim=-1
        )
        topk_idx = torch.topk(similarities, top_k).indices
        return self.memory_values[topk_idx]

    # -------------------------
    # Integration
    # -------------------------
    def integrate(self, core_output, study_output, query, top_k=5):
        """
        Fuse Core output, Study output, and retrieved memory context.
        core_output: Tensor [embed_dim]
        study_output: Tensor [embed_dim]
        query: Tensor [embed_dim]
        """
        memory_context = self.retrieve(query, top_k=top_k)
        mem_summary = memory_context.mean(dim=0)  # summary of retrieved memory

        combined = torch.cat([core_output, study_output, mem_summary], dim=-1)
        return self.fusion(combined)

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, core_output, study_output, query, top_k=5):
        """
        Wrapper for integration (Core + Study + Memory).
        """
        return self.integrate(core_output, study_output, query, top_k=top_k)
