"""AI-Fold v0.1 State Encoder/Decoder

This module provides the explicit StateEncoder and StateDecoder
required by the v0.1 architecture review to make the diffusion
target well-defined.

The StateEncoder maps a structured state (entities + relations) 
to a latent vector z_t ∈ R^d_Z.

The StateDecoder maps z_t back to predicted state components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from aifold.config import ModelConfig, EntityTypeConfig, RelationTypeConfig


class StateEncoder(nn.Module):
    """Encode a full AI system state into a latent vector z_t ∈ R^d_Z.
    
    The state consists of:
    - Entities: N entities with types, attributes, content
    - Relations: N×N relation matrix with types
    - Metadata: horizon, step, etc.
    
    Output: z_t [d_Z]
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_H = config.d_H
        self.d_P = config.d_P
        self.d_Z = config.d_Z
        
        # Reuse entity and pair encoders
        self.entity_encoder = nn.Sequential(
            nn.Linear(config.d_H, config.d_H * 2),
            nn.GELU(),
            nn.Linear(config.d_H * 2, config.d_H),
        )
        
        self.pair_encoder = nn.Sequential(
            nn.Linear(config.d_P, config.d_P * 2),
            nn.GELU(),
            nn.Linear(config.d_P * 2, config.d_P),
        )
        
        # Global pooling from entities
        self.entity_pool = nn.Sequential(
            nn.Linear(config.d_H, config.d_H * 2),
            nn.GELU(),
            nn.Linear(config.d_H * 2, config.d_Z // 2),
        )
        
        # Global pooling from pairs
        self.pair_pool = nn.Sequential(
            nn.Linear(config.d_P, config.d_P * 2),
            nn.GELU(),
            nn.Linear(config.d_P * 2, config.d_Z // 2),
        )
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.LayerNorm(config.d_Z),
            nn.Linear(config.d_Z, config.d_Z),
        )
        
        # Learnable metadata embeddings
        self.horizon_embed = nn.Embedding(8, config.d_Z // 4)  # T ∈ {1..8}
        self.step_embed = nn.Embedding(100, config.d_Z // 4)   # step index
    
    def forward(
        self,
        H: torch.Tensor,                    # [N, d_H] entity states
        P: torch.Tensor,                    # [N, N, d_P] pair states
        mask: Optional[torch.Tensor] = None, # [N] valid entities
        horizon: Optional[torch.Tensor] = None,  # [1] horizon
        step: Optional[torch.Tensor] = None,     # [1] step index
    ) -> torch.Tensor:
        """Encode state to latent z_t."""
        N = H.shape[0]
        
        if mask is None:
            mask = torch.ones(N, dtype=torch.bool, device=H.device)
        
        # Encode entities
        H_enc = self.entity_encoder(H)  # [N, d_H]
        H_enc = H_enc * mask.unsqueeze(-1).float()
        
        # Pool entities (mean)
        entity_count = mask.float().sum().clamp(min=1)
        H_pooled = H_enc.sum(dim=0) / entity_count  # [d_H]
        H_global = self.entity_pool(H_pooled)       # [d_Z/2]
        
        # Encode pairs
        P_enc = self.pair_encoder(P)  # [N, N, d_P]
        P_enc = P_enc * mask.unsqueeze(-1).float() * mask.unsqueeze(-2).float()
        
        # Pool pairs (mean over valid pairs)
        pair_count = (mask.unsqueeze(-1) * mask.unsqueeze(-2)).float().sum().clamp(min=1)
        P_pooled = P_enc.sum(dim=(0, 1)) / pair_count  # [d_P]
        P_global = self.pair_pool(P_pooled)            # [d_Z/2]
        
        # Combine
        z = torch.cat([H_global, P_global], dim=-1)  # [d_Z]
        
        # Add metadata
        if horizon is not None:
            z = z + self.horizon_embed(horizon.clamp(0, 7))
        if step is not None:
            z = z + self.step_embed(step.clamp(0, 99))
        
        # Final projection
        z = self.final_proj(z)
        
        return z


class StateDecoder(nn.Module):
    """Decode latent z_t back to predicted state components.
    
    Outputs:
    - Entity predictions: per-entity state changes
    - Pair predictions: per-pair relation changes
    - Action logits: next action prediction
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_Z = config.d_Z
        self.d_H = config.d_H
        self.d_P = config.d_P
        self.num_entity_types = EntityTypeConfig.NUM_TYPES
        self.num_relation_types = RelationTypeConfig.NUM_TYPES
        
        # Entity decoder
        self.entity_decoder = nn.Sequential(
            nn.Linear(config.d_Z, config.d_H * 2),
            nn.GELU(),
            nn.Linear(config.d_H * 2, config.d_H),
        )
        
        # Pair decoder
        self.pair_decoder = nn.Sequential(
            nn.Linear(config.d_Z, config.d_P * 2),
            nn.GELU(),
            nn.Linear(config.d_P * 2, config.d_P),
        )
        
        # Action decoder (from z_t)
        self.action_decoder = nn.Sequential(
            nn.Linear(config.d_Z, config.d_H * 2),
            nn.GELU(),
            nn.Linear(config.d_H * 2, RelationTypeConfig.NUM_TYPES),  # Predict action type
        )
        
        # Entity state change predictor
        self.entity_change = nn.Sequential(
            nn.Linear(config.d_Z + config.d_H, config.d_H * 2),
            nn.GELU(),
            nn.Linear(config.d_H * 2, config.d_H),
        )
        
        # Pair relation change predictor
        self.pair_change = nn.Sequential(
            nn.Linear(config.d_Z + config.d_P, config.d_P * 2),
            nn.GELU(),
            nn.Linear(config.d_P * 2, config.d_P),
        )
    
    def forward(
        self,
        z: torch.Tensor,                    # [d_Z] or [B, d_Z]
        H: Optional[torch.Tensor] = None,    # [N, d_H] current entities
        P: Optional[torch.Tensor] = None,    # [N, N, d_P] current pairs
    ) -> dict:
        """Decode latent to state predictions."""
        
        # Predict next entities
        H_next = self.entity_decoder(z)  # [d_H] or [B, d_H]
        
        # Predict next pairs
        P_next = self.pair_decoder(z)    # [d_P] or [B, d_P]
        
        # Predict action
        action_logits = self.action_decoder(z)  # [num_relation_types]
        
        outputs = {
            'H_next': H_next,
            'P_next': P_next,
            'action_logits': action_logits,
        }
        
        # If current state provided, predict changes
        if H is not None:
            # Expand z to match H [B, N, d_H]
            if z.dim() == 1:
                z_exp = z.unsqueeze(0).expand(H.shape[0], H.shape[1], -1)
            elif z.dim() == 2:
                z_exp = z.unsqueeze(1).expand(-1, H.shape[1], -1)
            else:
                z_exp = z
            
            entity_input = torch.cat([z_exp, H], dim=-1)
            H_change = self.entity_change(entity_input)
            outputs['H_change'] = H_change
        
        if P is not None:
            # z_exp needs to be [B, N, N, d_Z] to match P [B, N, N, d_P]
            if z.dim() == 1:
                z_exp = z.unsqueeze(0).unsqueeze(0).expand(P.shape[0], P.shape[1], P.shape[1], -1)
            elif z.dim() == 2:
                z_exp = z.unsqueeze(1).unsqueeze(1).expand(-1, P.shape[1], P.shape[1], -1)
            else:
                z_exp = z
            
            pair_input = torch.cat([z_exp, P], dim=-1)
            P_change = self.pair_change(pair_input)
            outputs['P_change'] = P_change
        
        return outputs


class StateAutoencoder(nn.Module):
    """Combined encoder-decoder for pretraining/supervision."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.encoder = StateEncoder(config)
        self.decoder = StateDecoder(config)
    
    def forward(
        self,
        H: torch.Tensor,
        P: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        horizon: Optional[torch.Tensor] = None,
        step: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Encode then decode state."""
        z = self.encoder(H, P, mask, horizon, step)
        recon = self.decoder(z, H, P)
        return z, recon
    
    def encode(self, H, P, mask=None, horizon=None, step=None):
        return self.encoder(H, P, mask, horizon, step)
    
    def decode(self, z, H=None, P=None):
        return self.decoder(z, H, P)