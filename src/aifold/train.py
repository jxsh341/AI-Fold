"""AI-Fold v0.1 Training and Experiment Runner"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, Optional
import json
import os
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm

from aifold.config import ModelConfig, ExperimentConfig, EXPERIMENTS
from aifold.model import AIModel, create_model_from_experiment
from aifold.data.dataset import create_dataloaders, generate_synthetic_data


@dataclass
class TrainingConfig:
    """Training configuration."""
    experiment: str = 'C_entity_pair'
    data_path: Optional[str] = None
    batch_size: int = 4
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    log_interval: int = 50
    eval_interval: int = 1000
    save_interval: int = 5000
    output_dir: str = './outputs'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    num_workers: int = 4
    
    # Loss weights
    diffusion_weight: float = 1.0
    action_weight: float = 1.0
    confidence_weight: float = 0.5
    state_weight: float = 0.5


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_optimizer(model: AIModel, config: TrainingConfig) -> optim.Optimizer:
    """Create optimizer with weight decay."""
    return optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def create_scheduler(optimizer: optim.Optimizer, config: TrainingConfig, num_training_steps: int):
    """Create learning rate scheduler with warmup."""
    
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        # Cosine decay
        progress = (step - config.warmup_steps) / max(1, num_training_steps - config.warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159))).item()
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_loss(model: AIModel, batch: Dict, config: TrainingConfig) -> tuple[torch.Tensor, Dict]:
    """Compute total loss for a batch."""
    
    # Forward pass
    outputs = model(
        type_ids=batch['type_ids'],
        attributes=batch.get('attributes'),
        content=batch.get('content'),
        content_mask=batch.get('content_mask'),
        relation_types=batch.get('relation_types'),
        temporal_offsets=batch.get('temporal_offsets'),
        causal_types=batch.get('causal_types'),
        structural_types=batch.get('structural_types'),
        mask=batch.get('mask'),
        target_z=batch.get('target_z'),
        target_actions=batch.get('target_actions'),
        target_success=batch.get('target_success'),
        target_entity_errors=batch.get('target_entity_errors'),
        target_pair_errors=batch.get('target_pair_errors'),
        horizon=batch.get('horizon'),
        goal_type=batch.get('goal_type'),
        num_samples=config.batch_size,  # Generate same number as batch
    )
    
    losses = {}
    total_loss = 0
    
    # Diffusion loss
    if 'diffusion_loss' in outputs:
        diff_loss = outputs['diffusion_loss']
        losses['diffusion_loss'] = diff_loss
        total_loss += config.diffusion_weight * diff_loss
    
    # Action prediction loss
    if 'action_logits' in outputs and batch.get('target_actions') is not None:
        action_logits = outputs['action_logits']  # [B, T, num_actions]
        target_actions = batch['target_actions']  # [B, T]
        action_loss = F.cross_entropy(
            action_logits.reshape(-1, action_logits.shape[-1]),
            target_actions.reshape(-1),
            ignore_index=-1
        )
        losses['action_loss'] = action_loss
        total_loss += config.action_weight * action_loss
    
    # Confidence loss
    if 'entity_confidence' in outputs and batch.get('target_entity_errors') is not None:
        conf_loss, conf_info = model.confidence_head.compute_loss(
            {
                'entity_confidence': outputs['entity_confidence'],
                'pair_confidence': outputs['pair_confidence'],
                'trajectory_score': outputs['trajectory_score'],
                'success_probability': outputs['success_probability'],
            },
            batch['target_z'],
            batch['target_actions'],
            batch['target_success'],
            batch['target_entity_errors'],
            batch['target_pair_errors'],
        )
        losses['confidence_loss'] = conf_loss
        total_loss += config.confidence_weight * conf_loss
        losses.update(conf_info)
    
    # State reconstruction loss
    if 'state_pred' in outputs and batch.get('target_z') is not None:
        state_pred = outputs['state_pred']
        if 'H_change' in state_pred and batch.get('target_z') is not None:
            # Could add state reconstruction loss here
            pass
    
    losses['total_loss'] = total_loss
    return total_loss, losses


def evaluate(model: AIModel, dataloader, config: TrainingConfig, device: str) -> Dict:
    """Evaluate model on validation set."""
    model.eval()
    total_losses = {}
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            _, losses = compute_loss(model, batch, config)
            
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v
            num_batches += 1
    
    # Average
    for k in total_losses:
        total_losses[k] /= num_batches
    
    model.train()
    return total_losses


def train(config: TrainingConfig):
    """Main training loop."""
    
    set_seed(config.seed)
    
    # Setup output directory
    output_dir = Path(config.output_dir) / config.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)
    
    # Create model
    model = create_model_from_experiment(config.experiment).to(config.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data
    if config.data_path and os.path.exists(config.data_path):
        dataloaders = create_dataloaders(
            config.data_path, 
            model.config, 
            config.batch_size, 
            config.num_workers
        )
    else:
        # Generate synthetic data
        print("Generating synthetic data...")
        synthetic_data = generate_synthetic_data(
            num_samples=1000,
            config=model.config
        )
        # Save synthetic data
        data_path = output_dir / 'synthetic_data.json'
        with open(data_path, 'w') as f:
            json.dump(synthetic_data, f)
        
        dataloaders = create_dataloaders(
            str(data_path),
            model.config,
            config.batch_size,
            config.num_workers
        )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    # Optimizer and scheduler
    optimizer = create_optimizer(model, config)
    num_training_steps = len(train_loader) * config.num_epochs
    scheduler = create_scheduler(optimizer, config, num_training_steps)
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_losses = {}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            batch = {k: v.to(config.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            optimizer.zero_grad()
            loss, losses = compute_loss(model, batch, config)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v
            
            global_step += 1
            
            # Logging
            if global_step % config.log_interval == 0:
                pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})
            
            # Evaluation
            if global_step % config.eval_interval == 0:
                val_losses = evaluate(model, val_loader, config, config.device)
                print(f"\nStep {global_step} Validation: {val_losses}")
                
                # Save best model
                if val_losses.get('total_loss', float('inf')) < best_val_loss:
                    best_val_loss = val_losses['total_loss']
                    torch.save({
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_losses': val_losses,
                    }, output_dir / 'best_model.pt')
            
            # Checkpoint
            if global_step % config.save_interval == 0:
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, output_dir / f'checkpoint_{global_step}.pt')
        
        # Epoch summary
        avg_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}
        print(f"Epoch {epoch+1} Average: {avg_losses}")
    
    # Final save
    torch.save({
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, output_dir / 'final_model.pt')
    
    print("Training complete!")


def run_experiment(experiment: str, config: Optional[TrainingConfig] = None):
    """Run a single experiment from the A→J sequence."""
    
    if config is None:
        config = TrainingConfig(experiment=experiment)
    else:
        config.experiment = experiment
    
    print(f"\n{'='*60}")
    print(f"Running Experiment {experiment}: {EXPERIMENTS.get(experiment, experiment)}")
    print(f"{'='*60}")
    
    train(config)
    
    print(f"\nExperiment {experiment} complete!")


def run_ablation_sequence(base_config: Optional[TrainingConfig] = None):
    """Run the full A→J ablation sequence."""
    
    if base_config is None:
        base_config = TrainingConfig()
    
    # Core experiments A-G
    core_experiments = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    
    for exp in core_experiments:
        run_experiment(exp, base_config)
    
    # Additional experiments H-J
    additional = ['H', 'I', 'J']
    for exp in additional:
        run_experiment(exp, base_config)
    
    print("\nAll experiments complete!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='AI-Fold v0.1 Training')
    parser.add_argument('--experiment', type=str, default='C', 
                       choices=list(EXPERIMENTS.keys()) + ['all'],
                       help='Experiment to run')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        experiment=EXPERIMENTS.get(args.experiment, args.experiment),
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )
    
    if args.experiment == 'all':
        run_ablation_sequence(config)
    else:
        run_experiment(config.experiment, config)