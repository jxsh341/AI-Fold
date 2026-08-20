"""E3: Latent Dimension Scaling (d_Z)
Tests how latent dimension affects reconstruction fidelity and downstream prediction.
"""

import sys
sys.path.insert(0, 'C:/Users/user/AI-Fold/src')

import torch
import json
from pathlib import Path

from aifold import create_model_from_experiment, AIModel
from aifold.config import ModelConfig

from experiments.base_experiment import BaseExperiment, create_synthetic_data, evaluate_model, train_model


class LatentDimExperiment(BaseExperiment):
    """Test how latent dimension d_Z affects reconstruction and prediction."""
    
    def __init__(self, output_dir='./experiment_results'):
        super().__init__('E3_latent_dim_scaling', output_dir)
    
    def create_configs(self):
        return [
            {'name': 'dZ_128', 'd_Z': 128, 'd_H': 256, 'd_P': 64},
            {'name': 'dZ_256', 'd_Z': 256, 'd_H': 384, 'd_P': 128},
            {'name': 'dZ_512', 'd_Z': 512, 'd_H': 384, 'd_P': 128},  # baseline
            {'name': 'dZ_768', 'd_Z': 768, 'd_H': 512, 'd_P': 128},
            {'name': 'dZ_1024', 'd_Z': 1024, 'd_H': 512, 'd_P': 256},
            {'name': 'dZ_2048', 'd_Z': 2048, 'd_H': 768, 'd_P': 256},
        ]
    
    def create_model(self, config: Dict[str, Any]) -> AIModel:
        """Create model with custom latent dimension."""
        model_config = ModelConfig(
            d_H=config.get('d_H', 384),
            d_P=config.get('d_P', 128),
            d_Z=config['d_Z'],
            num_relational_blocks=8,
            num_heads=16,
            num_diffusion_blocks=12,
            num_diffusion_heads=16,
            num_diffusion_steps=64,
            num_recycles=4,
            num_action_classes=23,
        )
        return AIModel(model_config)
    
    def run_single(self, config: Dict[str, Any], 
                   train_steps: int = 500,
                   eval_steps: int = 20) -> Dict[str, float]:
        """Run single d_Z configuration."""
        
        device = 'cpu'  # Use CPU for consistent comparison
        model = self.create_model(config)
        
        print(f"Testing {config['name']} (d_Z={config['d_Z']}, d_H={config.get('d_H')}, d_P={config.get('d_P')})...")
        
        # Training
        train_batch = create_synthetic_data(batch_size=4)
        train_metrics = train_model(model, train_batch, steps=train_steps)
        
        # Evaluation
        eval_metrics = evaluate_model(model, create_synthetic_data(batch_size=8), 'cpu')
        
        # State reconstruction test
        model.eval()
        with torch.no_grad():
            test_batch = create_synthetic_data(batch_size=8)
            H, P = model.encode_input(
                type_ids=test_batch['type_ids'],
                attributes=test_batch['attributes'],
                relation_types=test_batch['relation_types'],
                temporal_offsets=test_batch['temporal_offsets'],
                mask=test_batch['mask'],
            )
            H, P = model.forward_trunk(H, P, test_batch['mask'])
            
            # Encode state to latent
            z = model.encode_state(H, P, test_batch['mask'])
            
            # Decode back
            recon = model.decode_state(z, H, P)
            
            # Reconstruction losses
            H_recon = recon['H_next']
            P_recon = recon['P_next']
            
            h_recon_loss = F.mse_loss(H_recon, H).item()
            p_recon_loss = F.mse_loss(P_recon, P).item()
            
            # Param count
            total_params = sum(p.numel() for p in model.parameters())
        
        return {
            'd_Z': config['d_Z'],
            'd_H': config.get('d_H', 384),
            'd_P': config.get('d_P', 128),
            'params': total_params,
            'action_acc': eval_metrics.get('action_acc', 0),
            'action_loss': eval_metrics.get('action_loss', 0),
            'diffusion_loss': eval_metrics.get('diffusion_loss', 0),
            'train_loss': train_metrics.get('train_loss', 0),
            'H_recon_loss': h_recon_loss,
            'P_recon_loss': p_recon_loss,
            'total_recon_loss': h_recon_loss + p_recon_loss,
        }


def main():
    exp = LatentDimExperiment()
    configs = exp.create_configs()
    results = exp.run(configs, train_steps=500, eval_steps=20)
    
    # Print summary
    print(f"\n{'='*100}")
    print("E3 LATENT DIMENSION SCALING SUMMARY")
    print(f"{'='*100}")
    print(f"{'d_Z':>6s} | {'d_H':>4s} | {'d_P':>4s} | {'params':>8s} | {'acc':>6s} | {'act_loss':>8s} | {'diff_loss':>8s} | {'H_recon':>8s} | {'P_recon':>8s} | {'total_recon':>10s}")
    print(f"{'-'*100}")
    for r in exp.results:
        m = r.metrics
        print(f"{m['d_Z']:>6d} | {m['d_H']:>4d} | {m['d_P']:>4d} | {m['params']/1e6:>7.1f}M | "
              f"{m['action_acc']:>6.3f} | {m['action_loss']:>8.3f} | {m['diffusion_loss']:>8.3f} | "
              f"{m['H_recon_loss']:>8.4f} | {m['P_recon_loss']:>8.4f} | {m['total_recon_loss']:>10.4f}")


if __name__ == '__main__':
    main()