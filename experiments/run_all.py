"""Master experiment runner - runs all E1-E5 experiments."""

import sys
sys.path.insert(0, 'C:/Users/user/AI-Fold/src')

import torch
import json
import time
from pathlib import Path

# Import all experiment modules
from experiments.e1_relation_ablation import main as run_e1
from experiments.e2_graph_vs_pairformer import main as run_e2
from experiments.e3_latent_dim import main as run_e3
from experiments.e4_diffusion_vs_ar import main as run_e4
from experiments.e5_adaptive_recycling import main as run_e5


def run_all_experiments():
    """Run all ablation experiments E1-E5."""
    
    experiments = [
        ('E1: Relation Ablation', 'e1_relation_ablation', run_e1),
        ('E2: Graph Transformer vs PairFormer', 'e2_graph_vs_pairformer', run_e2),
        ('E3: Latent Dim Scaling', 'e3_latent_dim', run_e3),
        ('E4: Diffusion vs AR vs Flow', 'e4_diffusion_vs_ar', run_e4),
        ('E5: Adaptive Recycling', 'e5_adaptive_recycling', run_e5),
    ]
    
    all_results = {}
    total_start = time.time()
    
    for name, module_name, run_fn in experiments:
        print(f"\n{'='*80}")
        print(f"RUNNING {name}")
        print(f"{'='*80}")
        
        start = time.time()
        try:
            run_fn()
            duration = time.time() - start
            print(f"{name} completed in {duration:.1f}s")
        except Exception as e:
            print(f"{name} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    total_duration = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED IN {total_duration:.1f}s")
    print(f"{'='*80}")


if __name__ == '__main__':
    run_all_experiments()