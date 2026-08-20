"""E5: Adaptive Recycling
Tests uncertainty-guided adaptive recycling vs fixed recycling.
"""

import sys
sys.path.insert(0, 'C:/Users/user/AI-Fold/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path

from aifold import create_model_from_experiment, AIModel
from aifold.config import ModelConfig
from aifold.modules.confidence import ConfidenceHead

from experiments.base_experiment import BaseExperiment, create_synthetic_data, evaluate_model, train_model


class AdaptiveRecyclingTrunk(nn.Module):
    """Relational trunk with adaptive recycling based on confidence."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Main trunk blocks
        from aifold.modules.core import RelationalBlock
        self.blocks = nn.ModuleList([
            RelationalBlock(config) for _ in range(config.num_relational_blocks)
        ])
        
        # Confidence head for recycling decisions
        self.recycling_confidence = ConfidenceHead(config)
        
        # Threshold for uncertainty
        self.uncertainty_threshold = 0.5  # Will be tuned
        self.max_iters = config.get('max_recycles', 8)
        self.min_iters = config.get('min_recycles', 2)
    
    def forward(self, H, P, mask=None):
        """Single trunk pass."""
        for block in self.blocks:
            H, P = block(H, P, mask)
        return H, P
    
    def adaptive_recycle(self, H, P, mask=None):
        """Adaptive recycling based on confidence."""
        prev_H, prev_P = H, P
        
        for i in range(self.max_iters):
            H, P = self.forward(H, P, mask)
            
            if i >= self.min_iters:
                # Check confidence
                with torch.no_grad():
                    # Run confidence head on current state
                    # We need Z for confidence head, use current H/P as proxy
                    z_dummy = torch.zeros(H.shape[0], 1, self.config.d_Z, device=H.device)
                    conf = self.recycling_confidence(H, P, z_dummy, mask)
                    
                    # Mean pair confidence as uncertainty measure
                    pair_conf = conf['pair_confidence'].mean() if isinstance(conf['pair_confidence'], torch.Tensor) else 0
                    entity_conf = conf['entity_confidence'].mean() if isinstance(conf['entity_confidence'], torch.Tensor) else 0
                    traj_conf = conf['trajectory_score'].mean() if isinstance(conf['trajectory_score'], torch.Tensor) else 0
                    
                    avg_conf = (pair_conf + entity_conf + traj_conf) / 3
                    
                    if avg_conf > self.uncertainty_threshold:
                        # High confidence, stop recycling
                        break
        
        return H, P
    
    def recycle(self, H, P, mask=None):
        """Standard fixed recycling."""
        for _ in range(self.config.num_recycles):
            H, P = self.forward(H, P, mask)
        return H, P


class AdaptiveRecyclingModel(AIModel):
    """AIModel with adaptive recycling trunk."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Replace trunk with adaptive version
        self.trunk = AdaptiveRecyclingTrunk(config)
    
    def forward_trunk(self, H, P, mask=None):
        return self.trunk.adaptive_recycle(H, P, mask)
    
    def forward_trunk_fixed(self, H, P, mask=None):
        return self.trunk.recycle(H, P, mask)


class AdaptiveRecyclingExperiment(BaseExperiment):
    """Compare fixed vs adaptive recycling."""
    
    def __init__(self, output_dir='./experiment_results'):
        super().__init__('E5_adaptive_recycling', output_dir)
    
    def create_configs(self):
        return [
            {'name': 'fixed_0', 'recycles': 0, 'adaptive': False},
            {'name': 'fixed_2', 'recycles': 2, 'adaptive': False},
            {'name': 'fixed_4', 'recycles': 4, 'adaptive': False},  # baseline
            {'name': 'fixed_8', 'recycles': 8, 'adaptive': False},
            {'name': 'adaptive_t0.3', 'threshold': 0.3, 'adaptive': True, 'min_iters': 2, 'max_iters': 8},
            {'name': 'adaptive_t0.5', 'threshold': 0.5, 'adaptive': True, 'min_iters': 2, 'max_iters': 8},
            {'name': 'adaptive_t0.7', 'threshold': 0.7, 'adaptive': True, 'min_iters': 2, 'max_iters': 8},
            {'name': 'adaptive_t0.9', 'threshold': 0.9, 'adaptive': True, 'min_iters': 2, 'max_iters': 8},
        ]
    
    def create_model(self, config: Dict[str, Any]):
        """Create model with adaptive or fixed recycling."""
        model_config = ModelConfig(
            d_H=384, d_P=128, d_Z=512,
            num_relational_blocks=8, num_heads=16,
            num_diffusion_blocks=12, num_diffusion_heads=16,
            num_diffusion_steps=64, num_recycles=config.get('recycles', 4),
            num_action_classes=23,
        )
        
        model = AdaptiveRecyclingModel(model_config)
        
        if config.get('adaptive', False):
            model.trunk.uncertainty_threshold = config['threshold']
            model.trunk.min_iters = config.get('min_iters', 2)
            model.trunk.max_iters = config.get('max_iters', 8)
        
        return model
    
    def run_single(self, config: Dict[str, Any], 
                   train_steps: int = 500,
                   eval_steps: int = 20) -> Dict[str, float]:
        """Run single recycling configuration."""
        
        device = 'cpu'
        model = self.create_model(config).to(device)
        
        print(f"Testing {config['name']}...")
        
        # Training
        train_batch = create_synthetic_data(batch_size=4)
        train_metrics = train_model(model, train_batch, steps=train_steps)
        
        # Evaluation
        eval_batch = create_synthetic_data(batch_size=8)
        eval_metrics = evaluate_model(model, eval_batch, device)
        
        # Measure effective recycles (for adaptive)
        if config.get('adaptive', False):
            # Track how many iterations were actually used
            pass
        
        total_params = sum(p.numel() for p in model.parameters())
        
        return {
            'name': config['name'],
            'adaptive': config.get('adaptive', False),
            'threshold': config.get('threshold', 0),
            'recycles': config.get('recycles', 0),
            'params': total_params,
            'action_acc': eval_metrics.get('action_acc', 0),
            'action_loss': eval_metrics.get('action_loss', 0),
            'diffusion_loss': eval_metrics.get('diffusion_loss', 0),
            'train_loss': train_metrics.get('train_loss', 0),
        }


def main():
    exp = AdaptiveRecyclingExperiment()
    configs = exp.create_configs()
    results = exp.run(configs, train_steps=500, eval_steps=20)
    
    print(f"\n{'='*100}")
    print("E5 ADAPTIVE RECYCLING SUMMARY")
    print(f"{'='*100}")
    for r in exp.results:
        m = r.metrics
        mode = 'adaptive' if m['adaptive'] else 'fixed'
        thr = f"t={m['threshold']}" if m['adaptive'] else f"r={m['recycles']}"
        print(f"{m['name']:15s} | {mode:8s} {thr:>6s} | "
              f"acc={m.get('action_acc',0):.3f} | "
              f"act_loss={m.get('action_loss',0):.3f} | "
              f"diff_loss={m.get('diffusion_loss',0):.3f}")


if __name__ == '__main__':
    main()