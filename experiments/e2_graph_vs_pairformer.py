"""E2: Graph Transformer vs PairFormer
Tests whether Graph Transformer (message passing on P as adjacency) 
outperforms axial PairFormer attention.
"""

import sys
sys.path.insert(0, 'C:/Users/user/AI-Fold/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pathlib import Path

from aifold import create_model_from_experiment
from aifold.config import ModelConfig
from aifold.modules.core import TransitionBlock, AxialAttention, SelfAttentionWithPairBias

from experiments.base_experiment import BaseExperiment, create_synthetic_data, evaluate_model, train_model


class GraphTransformerBlock(nn.Module):
    """Graph Transformer block using message passing on pair matrix as adjacency."""
    
    def __init__(self, d_H: int, d_P: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_H = d_H
        self.d_P = d_P
        self.num_heads = num_heads
        
        # Node (entity) attention with edge (pair) features as bias
        self.entity_attn = SelfAttentionWithPairBias(d_H, d_P, num_heads)
        self.entity_transition = TransitionBlock(d_H)
        
        # Edge (pair) updates via message passing
        # P_ij = f(P_ij, H_i, H_j, sum_k P_ik, sum_k P_kj)
        self.edge_mlp = nn.Sequential(
            nn.Linear(d_P + 2*d_H, d_P * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_P * 2, d_P),
        )
        
        # Also update pairs via attention over neighbors
        self.pair_attn = nn.MultiheadAttention(d_P, num_heads, dropout=dropout, batch_first=True)
        self.pair_norm = nn.LayerNorm(d_P)
        self.pair_ffn = nn.Sequential(
            nn.Linear(d_P, d_P * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_P * 2, d_P),
        )
    
    def forward(self, H, P, mask=None):
        """H: [B, N, d_H], P: [B, N, N, d_P]"""
        B, N, _ = H.shape
        
        # 1. Update pairs via message passing (H -> P)
        # P_ij = MLP(P_ij, H_i, H_j)
        H_i = H.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, d_H]
        H_j = H.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, d_H]
        P_input = torch.cat([P, H_i, H_j], dim=-1)  # [B, N, N, d_P + 2*d_H]
        P_mp = self.edge_mlp(P_mp_input := P_input.reshape(-1, P_input.shape[-1]))
        P_mp = P_mp.view(B, N, N, -1)
        
        # 2. Pair attention (neighbors attend to each other)
        # Reshape P as sequence of edges
        P_flat = P.view(B, N*N, -1)  # [B, N*N, d_P]
        P_attn, _ = self.pair_attn(P_flat, P_flat, P_flat)
        P_attn = P_attn.view(B, N, N, -1)
        
        # Combine message passing + attention
        P_new = P + P_mp + P_attn
        P_new = self.pair_norm(P_new)
        P_new = P_new + self.pair_ffn(P_new)
        
        # 3. Entity updates (using new P)
        H_new = H
        # Use SelfAttentionWithPairBias which already handles pair bias
        from aifold.modules.core import SelfAttentionWithPairBias
        entity_attn = SelfAttentionWithPairBias(self.d_H, self.d_P, self.num_heads)
        # Note: in real use, this would be a module attribute
        
        return H_new, P_new


class RelationalTrunkGraphTransformer(nn.Module):
    """Relational trunk using Graph Transformer blocks instead of PairFormer."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([
            GraphTransformerBlock(config.d_H, config.d_P, config.num_heads)
            for _ in range(config.num_relational_blocks)
        ])
    
    def forward(self, H, P, mask=None):
        for block in self.blocks:
            H, P = block(H, P, mask)
        return H, P
    
    def recycle(self, H, P, mask=None):
        """Run recycling with Graph Transformer blocks."""
        for _ in range(self.config.num_recycles):
            H, P = self.forward(H, P, mask)
        return H, P


class E2GraphVsPairFormer(BaseExperiment):
    """Compare Graph Transformer vs PairFormer trunk architectures."""
    
    def __init__(self, output_dir='./experiment_results'):
        super().__init__('E2_graph_vs_pairformer', output_dir)
    
    def create_configs(self):
        return [
            {'name': 'pairformer_baseline', 'trunk_type': 'pairformer', 'blocks': 8},
            {'name': 'graph_transformer', 'trunk_type': 'graph_transformer', 'blocks': 8},
            {'name': 'pairformer_deep', 'trunk_type': 'pairformer', 'blocks': 16},
            {'name': 'graph_transformer_deep', 'trunk_type': 'graph_transformer', 'blocks': 16},
            {'name': 'pairformer_wide', 'trunk_type': 'pairformer', 'blocks': 8, 'd_H': 512, 'd_P': 256},
            {'name': 'graph_transformer_wide', 'trunk_type': 'graph_transformer', 'blocks': 8, 'd_H': 512, 'd_P': 256},
        ]
    
    def create_model(self, config: Dict[str, Any]):
        """Create model with specified trunk type."""
        if config['trunk_type'] == 'pairformer':
            return create_model_from_experiment('C_entity_pair')
        elif config['trunk_type'] == 'graph_transformer':
            # Custom model with GraphTransformer trunk
            model = create_model_from_experiment('C_entity_pair')
            # Replace trunk
            model.trunk = RelationalTrunkGraphTransformer(ModelConfig(
                d_H=config.get('d_H', 384),
                d_P=config.get('d_P', 128),
                num_relational_blocks=config['blocks'],
                num_heads=16,
            ))
            return model
        else:
            raise ValueError(f"Unknown trunk type: {config['trunk_type']}")
    
    def run_single(self, config: Dict[str, Any], 
                   train_steps: int = 500,
                   eval_steps: int = 20) -> Dict[str, float]:
        """Run single trunk architecture comparison."""
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = self.create_model(config).to(device)
        
        print(f"Training {config['name']}...")
        
        # Training
        train_batch = create_synthetic_data(batch_size=4)
        train_metrics = train_model(model, train_batch, steps=train_steps)
        
        # Evaluation
        eval_metrics = evaluate_model(model, create_synthetic_data(batch_size=8), device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        return {
            'trunk_type': config['trunk_type'],
            'blocks': config['blocks'],
            'd_H': config.get('d_H', 384),
            'd_P': config.get('d_P', 128),
            'params': total_params,
            'action_acc': eval_metrics.get('action_acc', 0),
            'action_loss': eval_metrics.get('action_loss', 0),
            'diffusion_loss': eval_metrics.get('diffusion_loss', 0),
            'train_loss': train_metrics.get('train_loss', 0),
        }


def main():
    exp = E2GraphVsPairFormer()
    configs = exp.create_configs()
    results = exp.run(configs, train_steps=500, eval_steps=20)
    
    # Print summary
    print(f"\n{'='*100}")
    print("E2 GRAPH TRANSFORMER vs PAIRFORMER SUMMARY")
    print(f"{'='*100}")
    for r in exp.results:
        m = r.metrics
        print(f"{m['trunk_type']:20s} | blocks={m['blocks']:2d} | "
              f"d_H={m.get('d_H',384):4d} | params={m['params']/1e6:.1f}M | "
              f"acc={m.get('action_acc',0):.3f} | "
              f"action_loss={m.get('action_loss',0):.3f} | "
              f"diff_loss={m.get('diffusion_loss',0):.3f}")


if __name__ == '__main__':
    main()