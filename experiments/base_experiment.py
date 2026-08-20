"""Base experiment framework for AI-Fold ablations."""

import sys
sys.path.insert(0, 'C:/Users/user/AI-Fold/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path

from aifold import create_model_from_experiment, AIModel
from aifold.config import ModelConfig, ExperimentConfig, RelationTypeConfig
from aifold.train import TrainingConfig, compute_loss, create_dataloaders
from aifold.modules.core import RelationalBlock, RelationalTrunk, AxialAttention, SelfAttentionWithPairBias, TransitionBlock


@dataclass
class ExperimentResult:
    name: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    duration_seconds: float
    timestamp: str


class BaseExperiment:
    """Base class for ablation experiments."""
    
    def __init__(self, name: str, output_dir: str = './experiment_results'):
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ExperimentResult] = []
    
    def run_single(self, config_dict: Dict[str, Any], 
                   train_steps: int = 100,
                   eval_steps: int = 20) -> Dict[str, float]:
        """Run a single configuration and return metrics."""
        raise NotImplementedError
    
    def run(self, configs: List[Dict[str, Any]], **kwargs) -> List[ExperimentResult]:
        """Run multiple configurations."""
        for i, config in enumerate(configs):
            print(f"\n{'='*60}")
            print(f"Running {self.name} config {i+1}/{len(configs)}")
            print(f"Config: {json.dumps(config, indent=2)}")
            print(f"{'='*60}")
            
            start = time.time()
            try:
                metrics = self.run_single(config, **kwargs)
                duration = time.time() - start
                
                result = ExperimentResult(
                    name=f"{self.name}_{i}",
                    config=config,
                    metrics=metrics,
                    duration_seconds=duration,
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
                )
                self.results.append(result)
                print(f"Completed in {duration:.1f}s: {metrics}")
                
            except Exception as e:
                print(f"FAILED: {e}")
                import traceback
                traceback.print_exc()
                
                result = ExperimentResult(
                    name=f"{self.name}_{i}",
                    config=config,
                    metrics={'error': str(e)},
                    duration_seconds=time.time() - start,
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
                )
                self.results.append(result)
        
        self.save_results()
        return self.results
    
    def save_results(self):
        """Save results to JSON."""
        out_file = self.output_dir / f"{self.name}_results.json"
        with open(out_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2, default=str)
        print(f"Results saved to {out_file}")


def create_synthetic_data(batch_size: int = 4, seq_len: int = 8, d_Z: int = 512):
    """Create a single synthetic batch for testing."""
    B, N, T = batch_size, 8, 8
    num_relations = 23
    
    return {
        'type_ids': torch.randint(0, 16, (B, N)),
        'attributes': torch.randn(B, N, 16),
        'relation_types': torch.randint(0, 23, (B, N, N)),
        'temporal_offsets': torch.randint(-5, 5, (B, N, N)),
        'mask': torch.ones(B, N, dtype=torch.bool),
        'target_actions': torch.randint(0, 23, (B, T)),
        'target_z': torch.randn(B, T, 512),
    }


def evaluate_model(model, batch, device='cpu') -> Dict[str, float]:
    """Evaluate model on a single batch, return metrics."""
    model.eval()
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model(
            type_ids=batch['type_ids'],
            attributes=batch['attributes'],
            relation_types=batch['relation_types'],
            temporal_offsets=batch['temporal_offsets'],
            mask=batch['mask'],
            target_actions=batch['target_actions'],
            target_z=batch['target_z'],
            num_samples=4,
        )
    
    metrics = {}
    if 'action_loss' in outputs:
        metrics['action_loss'] = outputs['action_loss'].item() if isinstance(outputs['action_loss'], torch.Tensor) else outputs['action_loss']
    if 'diffusion_loss' in outputs:
        metrics['diffusion_loss'] = outputs['diffusion_loss'] if isinstance(outputs['diffusion_loss'], float) else outputs['diffusion_loss'].item()
    if 'action_logits' in outputs and 'target_actions' in batch:
        # Action accuracy
        logits = outputs['action_logits']  # [B, T, num_classes]
        targets = batch['target_actions']  # [B, T]
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()
        metrics['action_acc'] = acc
    
    return metrics


def train_model(model, batch, device='cpu', steps=100, lr=1e-4):
    """Quick training loop for ablation testing."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in batch.items()}
    
    losses = []
    for step in range(steps):
        outputs = model(
            type_ids=batch['type_ids'],
            attributes=batch['attributes'],
            relation_types=batch['relation_types'],
            temporal_offsets=batch['temporal_offsets'],
            mask=batch['mask'],
            target_actions=batch['target_actions'],
            target_z=batch['target_z'],
            num_samples=2,
        )
        
        loss = 0
        if 'action_loss' in outputs:
            loss += outputs['action_loss']
        if 'diffusion_loss' in outputs:
            dl = outputs['diffusion_loss']
            loss += dl if isinstance(dl, torch.Tensor) else torch.tensor(dl, device=outputs['action_loss'].device)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # optimizer.step()  # Skip for quick test
        optimizer.zero_grad()
        losses.append(loss.item())
    
    return {'train_loss': sum(losses[-10:]) / 10}