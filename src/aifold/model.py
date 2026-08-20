"""AI-Fold v0.1 Main Model

Ties together all components into the complete v0.1 architecture.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from aifold.config import ModelConfig, ExperimentConfig
from aifold.modules.core import (
    EntityEncoder, PairConstructor, RelationalTrunk, RelationalBlock
)
from aifold.modules.state_codec import StateEncoder, StateDecoder, StateAutoencoder
from aifold.modules.diffusion import LatentDiffusionHead
from aifold.modules.confidence import ConfidenceHead, RankingHead


class AIModel(nn.Module):
    """Complete AI-Fold v0.1 model."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input encoding
        self.entity_encoder = EntityEncoder(config)
        self.pair_constructor = PairConstructor(config)
        
        # Relational trunk with recycling
        self.trunk = RelationalTrunk(config)
        
        # State codec (for diffusion targets)
        self.state_encoder = StateEncoder(config)
        self.state_decoder = StateDecoder(config)
        
        # Latent diffusion head
        self.diffusion_head = LatentDiffusionHead(config)
        
        # Confidence and ranking
        self.confidence_head = ConfidenceHead(config)
        self.ranking_head = RankingHead(config)
        
        # Action decoder from latent (per timestep)
        self.action_decoder = nn.Sequential(
            nn.Linear(config.d_Z, config.d_H * 2),
            nn.GELU(),
            nn.Linear(config.d_H * 2, config.num_action_classes),
        )
        
        # EMA model for inference
        self.ema_decay = 0.999
        self.ema_model = None
        self._init_ema()
    
    def _init_ema(self):
        """Initialize EMA model as a copy of main model."""
        import copy
        self.ema_model = copy.deepcopy(self)
        for param in self.ema_model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def update_ema(self):
        """Update EMA model parameters."""
        if self.ema_model is None:
            return
        for ema_param, param in zip(self.ema_model.parameters(), self.parameters()):
            ema_param.mul_(self.ema_decay).add_(param, alpha=1 - self.ema_decay)
        
        for ema_buf, buf in zip(self.ema_model.buffers(), self.buffers()):
            ema_buf.copy_(buf)
    
    def get_ema_model(self):
        """Get EMA model for inference."""
        if self.ema_model is None:
            self._init_ema()
        return self.ema_model
    
    def encode_input(
        self,
        type_ids: torch.Tensor,                    # [B, N]
        attributes: Optional[torch.Tensor] = None,  # [B, N, A]
        content: Optional[torch.Tensor] = None,     # [B, N, L, d_C]
        content_mask: Optional[torch.Tensor] = None,
        relation_types: Optional[torch.Tensor] = None,   # [B, N, N]
        temporal_offsets: Optional[torch.Tensor] = None, # [B, N, N]
        causal_types: Optional[torch.Tensor] = None,
        structural_types: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,             # [B, N]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode raw input to H, P."""
        
        B, N = type_ids.shape
        
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=type_ids.device)
        
        # Encode entities
        H = self.entity_encoder(
            type_ids.view(-1),
            attributes.view(-1, attributes.shape[-1]) if attributes is not None else None,
            content.view(-1, content.shape[-2], content.shape[-1]) if content is not None else None,
            content_mask.view(-1, content_mask.shape[-1]) if content_mask is not None else None,
        ).view(B, N, -1)
        
        # Construct pairs per sample (PairConstructor expects single-sample inputs)
        P_list = []
        for b in range(B):
            if relation_types is None:
                rt = torch.full((N, N), 
                              self.pair_constructor.relation_embed.num_embeddings - 1,
                              dtype=torch.long, device=type_ids.device)
            else:
                rt = relation_types[b]
            
            if temporal_offsets is None:
                to = torch.zeros(N, N, dtype=torch.long, device=type_ids.device)
            else:
                to = temporal_offsets[b]
            
            ct = causal_types[b] if causal_types is not None else None
            st = structural_types[b] if structural_types is not None else None
            
            P_b = self.pair_constructor(
                H[b], rt, to, ct, st
            )
            P_list.append(P_b)
        
        P = torch.stack(P_list, dim=0)  # [B, N, N, d_P]
        
        return H, P
    
    def forward_trunk(
        self,
        H: torch.Tensor,
        P: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run trunk with recycling."""
        return self.trunk.recycle(H, P, mask)
    
    def encode_state(
        self,
        H: torch.Tensor,
        P: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        horizon: Optional[torch.Tensor] = None,
        step: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode state to latent z_t."""
        return self.state_encoder(H, P, mask, horizon, step)
    
    def decode_state(
        self,
        z: torch.Tensor,
        H: Optional[torch.Tensor] = None,
        P: Optional[torch.Tensor] = None,
    ) -> dict:
        """Decode latent to state predictions."""
        return self.state_decoder(z, H, P)
    
    def generate_trajectories(
        self,
        H: torch.Tensor,
        P: torch.Tensor,
        horizon: Optional[torch.Tensor] = None,
        goal_type: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate M candidate future trajectories."""
        return self.diffusion_head.sample(
            H, P, horizon, goal_type, num_samples, mask=mask
        )
    
    def score_trajectories(
        self,
        H: torch.Tensor,
        P: torch.Tensor,
        Z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Score candidate trajectories with confidence head."""
        confidence = self.confidence_head(H, P, Z, mask)
        ranking = self.ranking_head(confidence)
        return {
            **confidence,
            'ranking_score': ranking,
        }
    
    def forward(
        self,
        # Input
        type_ids: torch.Tensor,
        attributes: Optional[torch.Tensor] = None,
        content: Optional[torch.Tensor] = None,
        content_mask: Optional[torch.Tensor] = None,
        relation_types: Optional[torch.Tensor] = None,
        temporal_offsets: Optional[torch.Tensor] = None,
        causal_types: Optional[torch.Tensor] = None,
        structural_types: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        
        # Target (for training)
        target_z: Optional[torch.Tensor] = None,        # [B, T, d_Z]
        target_actions: Optional[torch.Tensor] = None,   # [B, T]
        target_success: Optional[torch.Tensor] = None,   # [B]
        target_entity_errors: Optional[torch.Tensor] = None,
        target_pair_errors: Optional[torch.Tensor] = None,
        
        # Generation config
        horizon: Optional[torch.Tensor] = None,
        goal_type: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
    ) -> dict:
        """Full forward pass: encode -> trunk -> generate/score."""
        
        # Encode input
        H, P = self.encode_input(
            type_ids, attributes, content, content_mask,
            relation_types, temporal_offsets,
            causal_types, structural_types, mask
        )
        
        # Run trunk with recycling
        H, P = self.forward_trunk(H, P, mask)
        
        outputs = {
            'H': H,
            'P': P,
        }
        
        # Training mode: compute diffusion loss
        if target_z is not None:
            diff_loss, diff_info = self.diffusion_head.training_step(
                target_z, H, P, horizon, mask=mask
            )
            outputs['diffusion_loss'] = diff_loss
            outputs.update(diff_info)
        
        # Generate trajectories
        Z_candidates = self.generate_trajectories(
            H, P, horizon, goal_type, mask, num_samples
        )
        outputs['Z_candidates'] = Z_candidates  # [B, M, T, d_Z]
        
        # Score trajectories
        scored = self.score_trajectories(H, P, Z_candidates, mask)
        outputs.update(scored)
        
        # Decode top candidate to actions
        top_idx = self.ranking_head.get_top_k(scored, k=1)  # [B, 1]
        top_Z = torch.gather(Z_candidates, 1, top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, Z_candidates.shape[2], Z_candidates.shape[3]))
        top_Z = top_Z.squeeze(1)  # [B, T, d_Z]
        
        # Decode to actions (for all timesteps)
        action_logits = torch.stack([
            self.action_decoder(top_Z[:, t]) for t in range(top_Z.shape[1])
        ], dim=1)  # [B, T, num_action_classes]
        outputs['action_logits'] = action_logits
        
        # Action classification loss (direct supervision)
        if target_actions is not None:
            action_loss = F.cross_entropy(
                action_logits.reshape(-1, action_logits.shape[-1]),
                target_actions.reshape(-1),
                ignore_index=-1,
                label_smoothing=0.1
            )
            outputs['action_loss'] = action_loss
        
        # Also decode state predictions
        state_pred = self.decode_state(top_Z.mean(dim=1), H, P)
        outputs['state_pred'] = state_pred
        
        return outputs


def create_model_from_experiment(experiment_name: str) -> AIModel:
    """Create model configured for a specific experiment."""
    exp_config = ExperimentConfig(experiment=experiment_name)
    model_config = exp_config.get_model_config()
    return AIModel(model_config)


# Experiment aliases
EXPERIMENTS = {
    'A': 'A_flat_transformer',
    'B': 'B_entity_only',
    'C': 'C_entity_pair',
    'D': 'D_recycling',
    'E': 'E_diffusion',
    'F': 'F_confidence',
    'G': 'G_confidence_recycle',
    'H': 'H_trirel',
    'I': 'I_adaptive_recycling',
    'J': 'J_retrieval',
}