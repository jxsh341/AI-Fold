"""AI-Fold v0.1 Confidence Head

Implements the four confidence signals:
- Entity confidence (analogue of pLDDT)
- Pair confidence (analogue of PAE/PDE)
- Trajectory confidence (analogue of pTM/ipTM)
- Success probability (new for AI-Fold)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from aifold.config import ModelConfig
from aifold.modules.core import TransitionBlock, SelfAttentionWithPairBias, AxialAttention


class PairFormerBlock(nn.Module):
    """PairFormer block for confidence head (smaller version)."""
    
    def __init__(self, d_H: int, d_P: int, num_heads: int):
        super().__init__()
        # Pair updates
        self.pair_row_attn = AxialAttention(d_P, num_heads, axis='row')
        self.pair_col_attn = AxialAttention(d_P, num_heads, axis='col')
        self.pair_transition = TransitionBlock(d_P)
        
        # Entity updates
        self.entity_attn = SelfAttentionWithPairBias(d_H, d_P, num_heads)
        self.entity_transition = TransitionBlock(d_H)
    
    def forward(
        self,
        H: torch.Tensor,           # [N, d_H]
        P: torch.Tensor,           # [N, N, d_P]
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        P = self.pair_row_attn(P, mask)
        P = self.pair_col_attn(P, mask)
        P = self.pair_transition(P)
        
        H = self.entity_attn(H, P, mask)
        H = self.entity_transition(H)
        
        return H, P


class ConfidenceHead(nn.Module):
    """Confidence prediction head for AI-Fold trajectories.
    
    Outputs four confidence signals:
    1. entity_confidence: per-entity quality [N] (analogue of pLDDT)
    2. pair_confidence: per-pair quality [N, N] (analogue of PAE)
    3. trajectory_score: scalar trajectory quality (analogue of pTM)
    4. success_probability: binary success prediction
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_H = config.d_H
        self.d_P = config.d_P
        
        # Confidence trunk (small PairFormer stack)
        self.blocks = nn.ModuleList([
            PairFormerBlock(config.d_H, config.d_P, config.num_heads)
            for _ in range(config.num_confidence_blocks)
        ])
        
        # Entity confidence (pLDDT analogue): bins in [0, 100]
        self.entity_confidence_mlp = nn.Sequential(
            nn.LayerNorm(config.d_H),
            nn.Linear(config.d_H, config.d_H * 2),
            nn.GELU(),
            nn.Linear(config.d_H * 2, 50),  # 50 bins for [0, 100]
        )
        
        # Pair confidence (PAE analogue): bins in [0, 32] Å
        self.pair_confidence_mlp = nn.Sequential(
            nn.LayerNorm(config.d_P),
            nn.Linear(config.d_P, config.d_P * 2),
            nn.GELU(),
            nn.Linear(config.d_P * 2, 64),  # 64 bins
        )
        
        # Trajectory score (pTM analogue): global quality
        self.trajectory_score_mlp = nn.Sequential(
            nn.Linear(config.d_H + config.d_P, config.d_H),
            nn.GELU(),
            nn.Linear(config.d_H, 1),
            nn.Sigmoid(),
        )
        
        # Success probability
        self.success_mlp = nn.Sequential(
            nn.Linear(config.d_H + config.d_P, config.d_H),
            nn.GELU(),
            nn.Linear(config.d_H, 1),
            nn.Sigmoid(),
        )
        
        # Bin centers for expected value computation
        self.register_buffer('entity_bin_centers', 
                           torch.linspace(0.5, 99.5, 50) * 2.0)  # [0, 100]
        self.register_buffer('pair_bin_centers',
                           torch.linspace(0.25, 31.75, 64))  # [0, 32]
    
    def forward(
        self,
        H: torch.Tensor,                    # [B, N, d_H] trunk entities
        P: torch.Tensor,                    # [B, N, N, d_P] trunk pairs
        Z: torch.Tensor,                    # [B, M, T, d_Z] predicted trajectories
        mask: Optional[torch.Tensor] = None, # [B, N]
    ) -> dict:
        """Predict confidence signals for each candidate trajectory."""
        
        B, M, T, d_Z = Z.shape
        N = H.shape[1]
        
        # For each candidate, run confidence trunk
        # We can either: (a) run per-candidate, or (b) average over time
        # For efficiency, we pool Z over time and add to H/P
        
        Z_pool = Z.mean(dim=2)  # [B, M, d_Z]
        
        outputs = {
            'entity_confidence': [],
            'pair_confidence': [],
            'trajectory_score': [],
            'success_probability': [],
        }
        
        for m in range(M):
            # Combine trunk with trajectory-specific info
            # Z_pool: [B, M, d_Z] -> [B, 1, d_Z] -> expand to [B, N, d_Z] -> slice to [B, N, d_H]
            z_slice = Z_pool[:, m:m+1].expand(-1, N, -1)  # [B, N, d_Z]
            z_slice = z_slice[..., :self.d_H]  # Slice last dimension to d_H
            H_cand = H + z_slice
            
            # Simple projection if dimensions don't match
            if Z_pool.shape[-1] != self.d_H:
                H_cand = H
            
            # Run confidence trunk
            H_conf, P_conf = H_cand, P
            for block in self.blocks:
                H_conf, P_conf = block(H_conf, P_conf, mask)
            
            # Entity confidence
            entity_logits = self.entity_confidence_mlp(H_conf)  # [B, N, 50]
            entity_probs = F.softmax(entity_logits, dim=-1)
            entity_conf = (entity_probs * self.entity_bin_centers).sum(dim=-1)  # [B, N]
            outputs['entity_confidence'].append(entity_conf)
            
            # Pair confidence
            pair_logits = self.pair_confidence_mlp(P_conf)  # [B, N, N, 64]
            pair_probs = F.softmax(pair_logits, dim=-1)
            pair_conf = (pair_probs * self.pair_bin_centers).sum(dim=-1)  # [B, N, N]
            outputs['pair_confidence'].append(pair_conf)
            
            # Trajectory score (global)
            # Pool H_conf and P_conf
            H_pooled = H_conf.mean(dim=1)  # [B, d_H]
            P_pooled = P_conf.mean(dim=(1, 2))  # [B, d_P]
            global_feat = torch.cat([H_pooled, P_pooled], dim=-1)
            traj_score = self.trajectory_score_mlp(global_feat).squeeze(-1)
            outputs['trajectory_score'].append(traj_score)
            
            # Success probability
            succ_prob = self.success_mlp(global_feat).squeeze(-1)
            outputs['success_probability'].append(succ_prob)
        
        # Stack over candidates
        for key in outputs:
            outputs[key] = torch.stack(outputs[key], dim=1)  # [B, M, ...]
        
        return outputs
    
    def compute_loss(
        self,
        pred: dict,
        target_trajectory: torch.Tensor,     # [B, T, d_Z] ground truth
        target_actions: torch.Tensor,        # [B, T] ground truth actions
        target_success: torch.Tensor,        # [B] binary success
        target_entity_errors: torch.Tensor,  # [B, N] true entity errors
        target_pair_errors: torch.Tensor,    # [B, N, N] true pair errors
    ) -> tuple[torch.Tensor, dict]:
        """Compute confidence losses."""
        
        B, M = pred['entity_confidence'].shape[:2]
        
        # For now, compute loss against the best candidate (oracle)
        # In practice, would use the actual generated candidate's quality
        
        # Entity confidence loss (MSE against true errors)
        # target_entity_errors: [B, N] in [0, 100]
        entity_loss = 0
        for m in range(M):
            entity_loss += F.mse_loss(pred['entity_confidence'][:, m], target_entity_errors)
        entity_loss /= M
        
        # Pair confidence loss
        pair_loss = 0
        for m in range(M):
            pair_loss += F.mse_loss(pred['pair_confidence'][:, m], target_pair_errors)
        pair_loss /= M
        
        # Trajectory score loss
        # Compute true trajectory quality (e.g., negative action error)
        # For simplicity, use success as proxy
        traj_loss = 0
        for m in range(M):
            traj_loss += F.mse_loss(pred['trajectory_score'][:, m], target_success.float())
        traj_loss /= M
        
        # Success probability loss (BCE)
        succ_loss = 0
        for m in range(M):
            succ_loss += F.binary_cross_entropy(pred['success_probability'][:, m], target_success.float())
        succ_loss /= M
        
        total_loss = entity_loss + pair_loss + traj_loss + succ_loss
        
        return total_loss, {
            'entity_conf_loss': entity_loss.item(),
            'pair_conf_loss': pair_loss.item(),
            'traj_score_loss': traj_loss.item(),
            'success_loss': succ_loss.item(),
            'total_conf_loss': total_loss.item(),
        }


class RankingHead(nn.Module):
    """Rank candidates by confidence scores."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Ranking weights (learnable)
        self.entity_weight = nn.Parameter(torch.tensor(0.2))
        self.traj_weight = nn.Parameter(torch.tensor(0.5))
        self.succ_weight = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, confidence: dict) -> torch.Tensor:
        """Compute ranking score for each candidate.
        
        confidence: dict with entity_confidence [B, M, N],
                     pair_confidence [B, M, N, N],
                     trajectory_score [B, M],
                     success_probability [B, M]
        Returns: ranking_score [B, M]
        """
        
        # Mean entity confidence
        entity_score = confidence['entity_confidence'].mean(dim=-1)  # [B, M]
        
        # Trajectory score
        traj_score = confidence['trajectory_score']  # [B, M]
        
        # Success probability
        succ_score = confidence['success_probability']  # [B, M]
        
        # Weighted sum (weights are learnable parameters)
        ranking = (self.entity_weight * entity_score + 
                   self.traj_weight * traj_score + 
                   self.succ_weight * succ_score)
        
        return ranking
    
    def get_top_k(self, confidence: dict, k: int = 1) -> torch.Tensor:
        """Get indices of top-k candidates."""
        ranking = self.forward(confidence)
        _, indices = torch.topk(ranking, k, dim=1)
        return indices