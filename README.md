# AI-Fold

**An AlphaFold-inspired discovery engine for AI systems — plus an AF3-derived neural substrate for trajectory prediction.**

AI-Fold asks a different question than ordinary agent frameworks:

> *What AI system should exist, and how can we experimentally discover it?*

It does this in two layers:

1. **Discovery Engine** (`aifold_discovery/`) — evolves *candidate AI systems* (agent architectures: planners, memory, verifiers, tools, critics) through real experiments on live LLMs, using multi-dimensional fitness and evolutionary search. **Atropos** is absorbed as its laboratory substrate (environments, rollouts, rewards).
2. **Neural Substrate** (`src/aifold/`) — an AlphaFold 3-derived model that represents AI systems as typed relational graphs (entities + relations), refines them with recycling, and generates *distributions over future trajectories* via latent diffusion with confidence ranking.

The strict boundary: **RL optimizes candidates; AI-Fold evolves candidates.**

---

## Live Quickstart (real LLM, ~4 minutes)

```bash
pip install torch numpy tqdm

# Configure any OpenAI-compatible backend (NVIDIA NIM shown)
cat > .env <<'EOF'
AIFOLD_BASE_URL=https://integrate.api.nvidia.com/v1
AIFOLD_API_KEY=nvapi-...
AIFOLD_MODEL=meta/llama-3.1-8b-instruct
EOF

python run_live_discovery.py --generations 2 --pop-size 6 --max-calls 500
```

What you'll see:

```
[backend] openai-compat:https://integrate.api.nvidia.com/v1
    [exp] 85ced7e9 @ math.reasoning.v4: reasoning+0.75 ...
    [exp] a38a0996 @ generalization.heldout.v1: robustness-0.09 ...
          brittle under partial failure; enable retries/critic
[gen best] [85ced7e9] fitness=0.750 :: model=default
  [child 7c4035dd] <- ['3d9f02cc'] via 'conditional_router' (targeted: efficiency)

Mutation effectiveness (avg composite fitness delta):
  seed:add_tool_code        +0.0654     ← what the search LEARNED works
  rl_steps_300->450         -0.0958     ← ...and what doesn't
```

Backends auto-detected in order: `AIFOLD_BASE_URL` → `OPENAI_API_KEY` → `GROQ_API_KEY` → `NVAPI_KEY` → local Ollama/vLLM probes.

---

## Part I — The Discovery Engine

A candidate is a **genome**: the complete heritable specification of an AI system's structure.

```python
CandidateGenome(
    model    = ModelComponent(verifier_enabled=True),
    memory   = MemoryComponent(episodic_memory=True, working_memory_size=16),
    planning = PlanningComponent(decomposition=True, search_algorithm="beam"),
    tools    = ToolComponent(enabled_tools=["code"]),
    control  = ControlComponent(critic_enabled=True, router_type="conditional"),
)
```

Genes are **executable**, not descriptive. `run_live_discovery.py` turns each genome into real agent behavior:

| Gene | Real behavior at inference |
|------|---------------------------|
| `planning.decomposition` | plan-then-solve prompting |
| `search=beam/mcts` | parallel candidate solutions + majority vote |
| `verifier_enabled` | independent re-derivation pass; wrong answers repaired |
| `critic_enabled` | checklist critique gate with revision |
| `tools=["code"]` | model-written Python executed in a sandbox, output fed back |
| `memory.episodic` | cross-episode lesson scratchpad injected as context |
| `working_memory_size` | literally controls how much context gets assembled |

### Multi-dimensional fitness

Scalar RL reward becomes a capability profile over 9 axes:

```
reasoning        ######################  0.750
efficiency       ######################  0.750
robustness       #############           0.450
tool_use                                 n/a
```

This enables Pareto-front selection (capability specialists survive), novelty search in fitness space, and per-axis bottleneck diagnosis that *directs* mutation: weakest axis → targeted hypothesis → next experiment.

### The loop

```
seed population
   └─▶ SELECT parents (Pareto + novelty + uncertainty bonus)
        └─▶ MUTATE / CROSSOVER  (bottleneck-targeted hypotheses)
             └─▶ EXPERIMENT: candidate × environment  (live LLM calls)
                  └─▶ trajectories → evidence → FitnessVector update
                       └─▶ DIAGNOSE regressions, record scientific history
                            └─▶ RL-train top slice (Atropos hook)
                                 └─▶ CULL, repeat
```

### Environment registry (the experimental genome)

Each environment is versioned, typed metadata + a factory:

```python
EnvironmentSpec(
    registry_id="coding.execution.v3", family="coding",
    capability_axes=["coding"], difficulty="medium",
    reward_space="binary_correctness",
    constraints={"sandbox": "subprocess", "timeout_s": 10},
)
```

Shipped live environments — all scored by deterministic local verifiers, no judge model:

| Registry ID | Measures | Verification |
|---|---|---|
| `math.reasoning.v4` | reasoning | generated word problems, exact-match |
| `coding.execution.v3` | coding | unit tests **executed** in subprocess sandbox |
| `agent.selfcorrection.v2` | self_correction | trap problems; scores detect-and-recover |
| `memory.long_context.v2` | memory | needle-in-haystack where `working_memory_size` causally affects score |
| `generalization.heldout.v1` | generalization | hard-parameter held-out variants |

Registry entries can equally point at real **Atropos** `BaseEnv` classes (`env_factory=...`) — mock envs, live envs, and Atropos-native envs all share one adapter path (`atropos_bridge/adapter.py`).

### Scientific memory

Every experiment persists full provenance to `aifold_runs/experiments/all_records.jsonl`: genome snapshot → environment → trajectory group → per-axis fitness delta → failure diagnosis → mutation lineage. The archive answers *"which mutations actually work?"* from data, not intuition.

### Going fully live with Atropos RL

`LiveRLHook` archives top candidates as GRPO/SFT-ready JSON (honest `steps=0` — no weights touched without a GPU trainer). To train weights, implement `GrpoHook.train()` shelling into `atropos-main/example_trainer/grpo.py`, and register Atropos-native environments via `env_factory`.

---

## Part II — The Neural Substrate (AF3 transfer)

`src/aifold/` implements the AlphaFold 3 ideas that survived reverse-engineering, rebuilt for AI-system trajectories:

| Component | AF3 Analogue | Purpose |
|-----------|--------------|---------|
| **EntityEncoder** | target feat + atom conditioning | typed entities → H ∈ ℝ^(N×384) |
| **PairConstructor** | outer sum L(H_i)+R(H_j) | semantic/temporal/causal relations → P ∈ ℝ^(N×N×128) |
| **RelationalTrunk** | PairFormer ×48 | 8 blocks of pair row/col attention + pair-biased entity attention |
| **Recycling** | AF3 recycling | 4 passes, shared weights, zero-init injection |
| **StateEncoder/Decoder** | — | states ↔ z ∈ ℝ^512 (makes the diffusion target well-defined) |
| **LatentDiffusionHead** | EDM diffusion head | 64-step latent trajectory sampling, AdaLN-Zero, Fourier σ embeddings |
| **ConfidenceHead** | pLDDT/PAE/pTM | entity/pair/trajectory/success confidence |
| **RankingHead** | pTM ranking | M=8 candidates ranked, top-K returned |

Training extras: action cross-entropy with label smoothing, EMA weights (decay 0.999), gradient accumulation (4×), stochastic depth (0.1).

```bash
# Ablation sequence A→J (one hypothesis per run)
python run_aifold.py --experiment C --epochs 10   # entity+pair baseline
python run_aifold.py --experiment all             # full sequence
```

### Structural ablations E1–E5 (`experiments/`)

| Exp | Question answered |
|-----|-------------------|
| E1 | Which relation types are necessary? (temporal/causal/structural ablations) |
| E2 | PairFormer vs Graph Transformer trunk |
| E3 | Latent dimension scaling d_Z ∈ 128–2048 vs reconstruction fidelity |
| E4 | Diffusion vs autoregressive vs flow matching at fixed compute |
| E5 | Fixed vs confidence-guided adaptive recycling |

---

## Project Structure

```
AI-Fold/
├── run_live_discovery.py      # ★ LIVE evolution loop (LLM-backed)
├── .env                       # backend config (gitignored)
│
├── aifold_discovery/          # ── Discovery engine (absorbed Atropos) ──
│   ├── core/
│   │   ├── genome.py          #   CandidateGenome + components
│   │   ├── fitness.py         #   FitnessVector (9 axes, Pareto, novelty)
│   │   ├── candidate.py       #   Candidate + Population
│   │   └── experiment.py      #   TrajectoryEvidence, ExperimentRecord/Store
│   ├── atropos_bridge/
│   │   ├── registry.py        #   EnvironmentRegistry (experimental genome)
│   │   └── adapter.py         #   Mock + Atropos-native adapters (one path)
│   ├── live/
│   │   ├── backend.py         #   OpenAI-compat backends + auto-detection
│   │   ├── scaffolding.py     #   GenomeScaffold: genes → agent behavior
│   │   ├── environments.py    #   4 verifiable live environments
│   │   ├── registry_live.py   #   live registry wiring
│   │   └── trainer.py         #   LiveRLHook / GrpoHook boundary
│   ├── evaluation/fitness.py  #   evidence → axis attribution + diagnosis
│   ├── evolution/
│   │   ├── mutation.py        #   14 structure mutations, axis-targeted
│   │   └── selection.py       #   Pareto + novelty selection
│   ├── search/evolutionary.py #   DiscoveryEngine main loop
│   └── memory/archive.py      #   scientific history persistence
│
├── src/aifold/                # ── Neural substrate (AF3 transfer) ──
│   ├── model.py               #   AIModel end-to-end
│   ├── modules/{core,state_codec,diffusion,confidence}.py
│   ├── data/dataset.py        #   causal-masked loading
│   └── train.py               #   A→J runner
│
├── experiments/               # E1–E5 structural ablations
├── run_aifold.py              # neural training entry point
├── atropos-main/              # vendored Atropos reference (gitignored)
└── aifold_runs/               # run artifacts (gitignored)
```

## Key design decisions

1. **Evolution above RL** — genomes mutate structure; weights are never touched by the search. The boundary is enforced in code (`RLTrainHook`).
2. **Fitness vectors, not scalar reward** — Pareto preservation keeps specialists alive; bottleneck diagnosis directs mutation.
3. **Trajectories are scientific evidence** — nothing discarded: every rollout keeps provenance, delta, and diagnosis forever.
4. **One adapter path for all environments** — mock (CI), live (LLM), and Atropos-native envs share `collect_trajectories`.
5. **Latent diffusion over state latents** — continuous `z_t` avoids invalid Gaussian noise on token IDs; actions decode separately.
6. **Causal masking is a loader invariant** — future information cannot leak through P, retrieval, or attributes.
7. **TriRel stays an ablation** — triangular relational composition is config-gated until Experiment H shows it transfers.

## Status & honest caveats

- ✅ Live evolution verified on NVIDIA NIM (~65 calls / small sweep; scale linearly with population × generations)
- ✅ Neural substrate forward/backward/sampling verified; not yet trained to convergence on real trajectory corpora
- ⚠️ Free-tier NIM keys are rate-limited: use `--max-concurrency 2` on larger sweeps
- ⚠️ Weight-training hook requires GPU infrastructure (see `live/trainer.py::GrpoHook`)
- ⚠️ Fitness attribution currently uses pass-rate/variance heuristics; calibration is future work

## License

Apache 2.0 (matching AlphaFold 3 and Atropos).
