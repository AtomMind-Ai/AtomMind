"""
Defines SmallScientificLLM, a transformer-based model that integrates domain expert
networks, a general knowledge backbone, symbolic reasoning, optimization, and chat expert modules.
"""

from typing import Dict, Optional
import torch
from torch import nn, Tensor
from models.den import DomainExpertNetwork
from models.gkb import GeneralKnowledgeBackbone
from models.srm import SymbolicReasoningModule
from models.oam import OptimizationAlgorithmModule
from models.cen import ChatExpertNetwork
from config import (
    HIDDEN_SIZE, NUM_ATTENTION_HEADS,
    DEN_LAYERS, GKB_LAYERS, SRM_LAYERS, OAM_LAYERS, DOMAINS
)


class SmallScientificLLM(nn.Module):
    """
    SmallScientificLLM integrates multiple modules for processing scientific text.

    Modules:
        - Domain Expert Networks (one per domain)
        - General Knowledge Backbone
        - Symbolic Reasoning Module
        - Optimization Algorithm Module
        - Chat Expert Network
        - Output linear projection

    Args:
        None
    """

    def __init__(self) -> None:
        super().__init__()
        self.dens = nn.ModuleDict({
            domain: DomainExpertNetwork(HIDDEN_SIZE, DEN_LAYERS, NUM_ATTENTION_HEADS)
            for domain in DOMAINS
        })
        self.gkb = GeneralKnowledgeBackbone(HIDDEN_SIZE, GKB_LAYERS, NUM_ATTENTION_HEADS)
        self.srm = SymbolicReasoningModule(HIDDEN_SIZE, SRM_LAYERS, NUM_ATTENTION_HEADS)
        self.oam = OptimizationAlgorithmModule(HIDDEN_SIZE, OAM_LAYERS, NUM_ATTENTION_HEADS)
        self.cen = ChatExpertNetwork(HIDDEN_SIZE, CEN_LAYERS, NUM_ATTENTION_HEADS)
        self.output_layer = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

    def forward(self, x_dict: Dict[str, Tensor], chat_tensor: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through the SmallScientificLLM.

        Args:
            x_dict (Dict[str, Tensor]): Input tensors for each domain.
            chat_tensor (Optional[Tensor]): Optional chat input tensor.

        Returns:
            Tensor: Model output tensor of shape [batch, seq_len, hidden_size].
        """
        den_outputs = [self.dens[d](x_dict[d]) for d in DOMAINS]
        gkb_out = self.gkb(den_outputs)
        srm_out = self.srm(gkb_out)
        oam_out = self.oam(srm_out)

        if chat_tensor is not None:
            chat_out = self.cen(chat_tensor)
            final_out = torch.cat([oam_out, chat_out], dim=1)
        else:
            final_out = oam_out

        return self.output_layer(final_out)
