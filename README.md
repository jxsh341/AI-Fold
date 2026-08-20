# AI-Fold v0.1

**An AlphaFold-inspired system for AI trajectory prediction.**

AI-Fold applies the architectural principles of AlphaFold 3 — structured entity/pair representations, relation-biased attention, iterative recycling, conditional generative sampling, multiple hypotheses, and confidence estimation — to the problem of predicting future trajectories of AI systems.

## Architecture Overview

AI-Fold models an AI system as a **temporal, typed relational graph**:

```
AI System State S_t = (H_t, P_t, G_t, C_t, U_t)
  H_t ∈ ℝ^(N×384)   : Entity states (agents, tools, goals, observations, ...)
  P_t ∈ ℝ^(N×N×128) : Pairwise relations (uses, depends_on, causes, temporal, ...)
  G_t               : Global/system state
  C_t               : Retrieved context (memories, exemplar trajectories)
  U_t               : Uncertainty state
```

### Core Components

| Component | AF3 Analogue | Purpose |
|-----------|--------------|---------|
| **EntityEncoder** | MSA + target feat | Encode typed entities → H [N, 384] |
| **PairConstructor** | Outer sum (L+R) | Build initial relations P [N, N, 128] |
| **RelationalTrunk** | PairFormer × 48 | Recursive H/P refinement with pair-biased attention |
| **Recycling** | AF3 recycling | 4 iterations of trunk with shared weights |
| **StateEncoder/Decoder** | — | Encode/decode states to latent z ∈ ℝ^512 |
| **LatentDiffusionHead** | Diffusion Head | Generate M=8 candidate future trajectories |
| **ConfidenceHead** | Confidence Head | Entity/pair/trajectory/success confidence |
| **RankingHead** | pTM/ipTM ranking | Select top-K candidates |

## Experiment Sequence (A→J)

Each experiment isolates one architectural hypothesis:

| Exp | Name | Tests |
|-----|------|-------|
| A | Flat Transformer | Baseline: standard next-action prediction |
| B | Entity Only | Single-stream entity representation |
| C | Entity + Pair | Dual-stream H+P with pair-biased attention |
| D | Recycling | Iterative trunk refinement |
| E | Latent Diffusion | Generative trajectory sampling |
| F | Confidence | Confidence heads + ranking |
| G | Confidence Recycle | Confidence-guided refinement |
| H | TriRel Ablation | 3-way relational reasoning |
| I | Adaptive Recycling | Learned stopping criterion |
| J | Retrieval | External context integration |

Run sequentially: `python run_aifold.py --experiment all`

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Run Experiment C (entity + pair) with synthetic data
python run_aifold.py --experiment C --epochs 10

# Run full ablation sequence
python run_aifold.py --experiment all

# Run with custom data
python run_aifold.py --experiment E --data_path ./trajectories.json --epochs 50
```

## Data Format

Training data uses a typed JSON format:

```json
{
  "id": "traj_001",
  "split": "train",
  "type_ids": [0, 2, 5, 1],  // AGENT, TASK, TOOL, GOAL
  "attributes": [[...], [...], [...], [...]],
  "relation_types": [[...]],
  "temporal_offsets": [[...]],
  "target_z": [[...]],  // Pre-encoded latent trajectory [T, 512]
  "target_actions": [0, 1, 2],
  "target_success": true,
  "horizon": 8
}
```

See `src/aifold/data/dataset.py` for full schema.

## Key Design Decisions

1. **Latent diffusion, not token diffusion** — AF3 diffuses continuous 3D coordinates; AI-Fold diffuses continuous state latents `z_t ∈ ℝ^512`, then decodes to actions.

2. **Explicit StateEncoder/Decoder** — Makes diffusion target well-defined with reconstruction supervision.

3. **Causal masking enforced in data loader** — Prevents temporal leakage; hard invariant, not convention.

4. **Retrieval as pluggable subsystem** — Disabled (R=0) for core experiments A-G; Experiment J adds it.

5. **Parameterized diffusion steps** — Default 32 (not AF3's 200); ablate 16/32/64/128/200.

6. **TriRel is an ablation (Experiment H)** — Not in v0.1 core; triangular reasoning may or may not transfer.

## Project Structure

```
AI-Fold/
├── run_aifold.py           # Entry point
├── requirements.txt
├── src/
│   └── aifold/
│       ├── config.py       # All configurations
│       ├── model.py        # Main AIModel
│       ├── train.py        # Training loop
│       ├── modules/
│       │   ├── core.py         # EntityEncoder, PairConstructor, RelationalTrunk
│       │   ├── state_codec.py  # StateEncoder/Decoder
│       │   ├── diffusion.py    # LatentDiffusionHead
│       │   └── confidence.py   # ConfidenceHead, RankingHead
│       └── data/
│           └── dataset.py      # Dataset, collation, causal masking
└── outputs/                # Checkpoints and logs
```

## Citation

If you use AI-Fold in your research, please cite:

```bibtex
@misc{aifold2026,
  title={AI-Fold: An AlphaFold-Inspired Architecture for AI Trajectory Prediction},
  author={...},
  year={2026}
}
```

## License

Apache 2.0 (matching AlphaFold 3)