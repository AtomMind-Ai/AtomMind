"""
orchestrator.py

Coordinates Core SLM, Study Net, and Memory System into a unified pipeline.
Acts as the "brain conductor": routes queries, merges outputs,
and returns coherent responses.
"""

import torch
import torch.nn as nn
from memory.memory_system import MemorySystem


class Orchestrator(nn.Module):
    def __init__(self, core_model, study_model, tokenizer, embed_dim=768, device="cuda"):
        """
        Args:
            core_model: Core SLM (general reasoning model)
            study_model: Study Net SLM (domain-specific model)
            tokenizer: Shared tokenizer
            embed_dim: Dimension for embeddings
            device: "cuda" or "cpu"
        """
        super().__init__()
        self.core = core_model.to(device)
        self.study = study_model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        # Memory system
        self.memory = MemorySystem(embed_dim=embed_dim, memory_size=2048, device=device)

        # Final projection to logits (vocab space)
        self.output_layer = nn.Linear(embed_dim, tokenizer.vocab_size).to(device)

    def encode(self, text):
        """Tokenize and get hidden embeddings (shared encoder logic)."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            core_hidden = self.core(**inputs).last_hidden_state.mean(dim=1).squeeze(0)
            study_hidden = self.study(**inputs).last_hidden_state.mean(dim=1).squeeze(0)
        return core_hidden, study_hidden

    def forward(self, query_text, top_k=5):
        """
        Run orchestrator on a query.
        - Encodes text into Core + Study embeddings
        - Retrieves memory
        - Integrates everything
        - Returns logits over vocab
        """
        # Encode query
        core_out, study_out = self.encode(query_text)
        query_embed = (core_out + study_out) / 2  # blended query rep

        # Store into memory
        self.memory.store(query_embed, study_out)

        # Fuse via memory
        integrated = self.memory(core_out, study_out, query_embed, top_k=top_k)

        # Map to vocabulary logits
        logits = self.output_layer(integrated)
        return logits

    def generate(self, query_text, max_new_tokens=50, top_k=5):
        """
        Generate a response using orchestrator.
        """
        logits = self.forward(query_text, top_k=top_k)
        probs = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, 1).item()
        generated_ids = [next_token_id]

        # greedy / autoregressive loop
        for _ in range(max_new_tokens - 1):
            token = self.tokenizer.decode([generated_ids[-1]])
            core_out, study_out = self.encode(token)
            query_embed = (core_out + study_out) / 2
            self.memory.store(query_embed, study_out)

            integrated = self.memory(core_out, study_out, query_embed, top_k=top_k)
            logits = self.output_layer(integrated)
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()
            generated_ids.append(next_token_id)

        return self.tokenizer.decode(generated_ids)
