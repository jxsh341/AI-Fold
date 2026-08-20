"""E4: Diffusion vs Autoregressive vs Flow Matching
Compares generative approaches at fixed compute budget.
"""

import sys
sys.path.insert(0, 'C:/Users/user/AI-Fold/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path

from aifold import create_model_from_experiment
from aifold.config import ModelConfig
from aifold.modules.diffusion import LatentDiffusionHead

from experiments.base_experiment import BaseExperiment, create_synthetic_data, evaluate_model, train_model


class AutoregressiveDecoder(nn.Module):
    """Autoregressive trajectory decoder."""
    
    def __init__(self, d_Z: int, d_H: int, horizon: int, num_actions: int, num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        self.d_Z = d_Z
        self.horizon = horizon
        self.num_actions = num_actions
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(d_H + 128, d_Z),  # H + P pooled
            nn.LayerNorm(d_Z),
        )
        
        # Transformer for autoregressive generation
        self.embedding = nn.Embedding(num_actions + 1, d_Z)  # +1 for BOS
        self.pos_embed = nn.Embedding(horizon + 1, d_Z)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_Z, nhead=8, dim_feedforward=d_Z*4,
            dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        
        self.action_head = nn.Linear(d_Z, num_actions)
        self.bos_token = num_actions  # special token
    
    def forward(self, H, P, target_actions=None):
        B, N, _ = H.shape
        
        # Pool H and P
        H_pool = H.mean(dim=1)  # [B, d_H]
        P_pool = P.mean(dim=(1, 2))  # [B, d_P]
        state = torch.cat([H_pool, P_pool], dim=-1)  # [B, d_H + d_P]
        
        context = self.state_encoder(state)  # [B, d_Z]
        context = context.unsqueeze(1)  # [B, 1, d_Z]
        
        if self.training and target_actions is not None:
            # Teacher forcing
            T = target_actions.shape[1]
            # BOS + target actions shifted
            inp = torch.cat([
                torch.full((target_actions.shape[0], 1), self.bos_token, device=target_actions.device, dtype=torch.long),
                target_actions[:, :-1]
            ], dim=1)  # [B, T]
            
            x = self.embedding(inp) + self.pos_embed(torch.arange(T+1, device=inp.device).unsqueeze(0))
            
            # Causal mask
            causal_mask = torch.triu(torch.ones(T+1, T+1, device=inp.device), diagonal=1).bool()
            
            out = self.decoder(tgt=x, memory=context, tgt_mask=causal_mask)
            logits = self.action_head(out)
            return logits[:, :-1]  # Predict next T actions
        else:
            # Inference: autoregressive generation
            return self.generate(context)
    
    def generate(self, context, num_samples=1):
        B = context.shape[0]
        # Simplified: just return context for now
        # Full autoregressive gen would be iterative
        return context.expand(-1, self.horizon, -1)


class FlowMatchingHead(nn.Module):
    """Flow Matching head (continuous normalizing flow)."""
    
    def __init__(self, d_Z: int, d_H: int, d_P: int, num_blocks: int = 6):
        super().__init__()
        # Vector field network: v_t(z_t, t) = z_target - z_t (approximately)
        self.net = nn.Sequential(
            nn.Linear(d_Z + 1 + 384 + 128, d_Z * 2),
            nn.GELU(),
            nn.Linear(d_Z * 2, d_Z * 2),
            nn.GELU(),
            nn.Linear(d_Z * 2, d_Z),
        )
        self.d_Z = d_Z
    
    def forward(self, z, t, H, P):
        B, N, _ = H.shape
        H_pool = H.mean(dim=1)
        P_pool = P.mean(dim=(1, 2))
        cond = torch.cat([H_pool, P_pool], dim=-1)
        
        # Input: [z, t, cond]
        t_expanded = t.view(-1, 1).expand(B, -1) if t.dim() == 1 else t
        inp = torch.cat([z, t_expanded.unsqueeze(-1), cond], dim=-1)
        return self.net(inp)
    
    def sample(self, H, P, num_steps=50, num_samples=1):
        B = H.shape[0]
        z = torch.randn(B, num_samples, self.d_Z, device=H.device)
        
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B, num_samples), i * dt, device=H.device)
            v = self.forward(z, t, H, P)
            z = z + v * dt
        return z


class DiffusionVsARvsFlow(BaseExperiment):
    """Compare Diffusion vs Autoregressive vs Flow Matching."""
    
    def __init__(self, output_dir='./experiment_results'):
        super().__init__('E4_diffusion_vs_ar_vs_flow', output_dir)
    
    def create_configs(self):
        return [
            {'name': 'diffusion', 'type': 'diffusion', 'steps': 64},
            {'name': 'diffusion_32', 'type': 'diffusion', 'steps': 32},
            {'name': 'autoregressive', 'type': 'autoregressive'},
            {'name': 'flow_matching', 'type': 'flow_matching', 'steps': 50},
        ]
    
    def create_model(self, config: Dict[str, Any]):
        """Create model with specified generative head."""
        model = create_model_from_experiment('C_entity_pair')
        
        if config['type'] == 'autoregressive':
            # Replace diffusion head with AR decoder
            model.diffusion_head = AutoregressiveDecoder(
                d_Z=model.config.d_Z,
                d_H=model.config.d_H,
                horizon=model.config.horizon_T,
                num_actions=model.config.num_action_classes,
            )
        elif config['type'] == 'flow_matching':
            # Replace with flow matching head
            model.diffusion_head = FlowMatchingHead(
                d_Z=model.config.d_Z,
                d_H=model.config.d_H,
                d_P=model.config.d_P,
            )
        # else: keep diffusion head
        
        return model
    
    def run_single(self, config: Dict[str, Any], 
                   train_steps: int = 500,
                   eval_steps: int = 20) -> Dict[str, float]:
        """Run single generative model comparison."""
        
        device = 'cpu'
        model = self.create_model(config).to(device)
        
        print(f"Testing {config['name']}...")
        
        # Training
        train_batch = create_synthetic_data(batch_size=4)
        train_metrics = train_model(model, train_batch, steps=train_steps)
        
        # Evaluation
        eval_batch = create_synthetic_data(batch_size=8)
        eval_metrics = evaluate_model(model, eval_batch, device)
        
        # Sample quality metrics
        model.eval()
        with torch.no_grad():
            # Generate samples
            H, P = model.encode_input(
                type_ids=eval_batch['type_ids'],
                attributes=eval_batch['attributes'],
                relation_types=eval_batch['relation_types'],
                temporal_offsets=eval_batch['temporal_offsets'],
                mask=eval_batch['mask'],
            )
            H, P = model.forward_trunk(H, P, eval_batch['mask'])
            
            if config['type'] == 'diffusion':
                Z = model.diffusion_head.sample(H, P, num_samples=8)
            elif config['type'] == 'autoregressive':
                Z = model.diffusion_head(H, P)  # AR generation
            elif config['type'] == 'flow_matching':
                Z = model.diffusion_head.sample(H, P, num_samples=8)
            
            # Diversity: pairwise distance between samples
            if Z.dim() == 4:  # [B, M, T, d_Z]
                M = Z.shape[1]
                diversity = 0
                for i in range(M):
                    for j in range(i+1, M):
                        diversity += F.mse_loss(Z[:, i], Z[:, j]).item()
                diversity /= (M * (M - 1) / 2)
            else:
                diversity = 0
        
        total_params = sum(p.numel() for p in model.parameters())
        
        return {
            'type': config['type'],
            'steps': config.get('steps', 'N/A'),
            'params': total_params,
            'action_acc': eval_metrics.get('action_acc', 0),
            'action_loss': eval_metrics.get('action_loss', 0),
            'diffusion_loss': eval_metrics.get('diffusion_loss', 0),
            'train_loss': train_metrics.get('train_loss', 0),
            'diversity': diversity,
        }


def main():
    exp = DiffusionVsARvsFlow()
    configs = exp.create_configs()
    results = exp.run(configs, train_steps=500, eval_steps=20)
    
    print(f"\n{'='*100}")
    print("E4 DIFFUSION vs AUTOREGRESSIVE vs FLOW MATCHING SUMMARY")
    print(f"{'='*100}")
    for r in exp.results:
        m = r.metrics
        print(f"{m['type']:15s} | steps={str(m['steps']):>5s} | params={m['params']/1e6:>5.1f}M | "
              f"acc={m.get('action_acc',0):.3f} | act_loss={m.get('action_loss',0):.3f} | "
              f"diff_loss={m.get('diffusion_loss',0):.3f} | div={m.get('diversity',0):.4f}")


if __name__ == '__main__':
    main()