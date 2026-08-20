#!/usr/bin/env python3
"""AI-Fold v0.1 Entry Point

AlphaFold-inspired system for AI trajectory prediction.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from aifold.train import run_experiment, run_ablation_sequence, TrainingConfig
from aifold.config import EXPERIMENTS


def main():
    parser = argparse.ArgumentParser(
        description='AI-Fold v0.1 - AlphaFold-inspired AI trajectory prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available experiments:
{chr(10).join(f"  {k}: {v}" for k, v in EXPERIMENTS.items())}

Examples:
  python run_aifold.py --experiment C --epochs 50
  python run_aifold.py --experiment all --data_path ./data/trajectories.json
  python run_aifold.py --experiment E --batch_size 8 --lr 5e-5
        """
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        default='C',
        choices=list(EXPERIMENTS.keys()) + ['all'],
        help='Experiment to run (A-J, or "all" for full ablation sequence)'
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Path to trajectory data JSON (if not provided, uses synthetic data)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='Output directory for checkpoints and logs'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if __import__('torch').cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Data loader workers'
    )
    
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
        num_workers=args.num_workers,
    )
    
    if args.experiment == 'all':
        run_ablation_sequence(config)
    else:
        run_experiment(config.experiment, config)


if __name__ == '__main__':
    main()