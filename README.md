# AI-Fold

**An AlphaFold-inspired discovery engine for AI systems — plus an AF3-derived neural substrate for trajectory prediction.**

> **Read this first:** every live result below is *signal, not a powered study*
> (n≈1–6 evaluated lineages per mutation). What the runs establish is that the
> loop closes end-to-end and — more importantly — that its measurement process
> survives adversarial probing. Four separate bugs that would have manufactured
> false discoveries were caught by interrogating our own positive results.
> A powered version = ≥10 lineages per mutation × 3 seeds (~5k calls/mutation-class).

## The main live finding is about measurement integrity

Across six sweeps (~1,700 real LLM calls, zero unhandled failures), probing
each apparent success exposed — and fixed — four distinct failure modes that
silently corrupt evolutionary attribution:

| # | Bug | How it was caught | Fix |
|---|-----|-------------------|-----|
| 1 | **Silent genes**: "winning" mutations (`rl_steps`, router-as-first-implemented) changed nothing at inference | grep proved no scaffold code path consumed them | phenotype contract: mutations tagged `live`/`rl`; `rl` unsampleable without a weight trainer |
| 2 | **Partial-composite bias**: child vs parent compared different axis sets mid-evaluation | every delta biased negative regardless of architecture | vs-parent computed on *common measured axes* only |
| 3 | **Capability blindness + lock-in**: uniformly-failing groups discarded as "zero information" → weakest axis invisible → singleton targeted-pool chose one gene 15× while **12 of 13** were never sampled | survivor audit: all gen-3 genes descended from gen-0's two router children | uniform groups retained (maximal weakness evidence); targeted-fallback + anti-repeat rules |
| 4 | **Coverage dodging**: offspring never measured on its inherited weak axis outranked the parent (0.835 "vs" 0.543) | the winning gene was silent for the axes it improved on | inheritance priors: unmeasured axes imputed at the *parent's* value — dodging ties, fixing wins |

**Bug #4 is worth isolating because it is not really about this repo.** Any
system that ranks candidates by scores aggregated over partially-measured
dimensions faces it: multi-objective evolutionary search, model selection
with per-benchmark evals, RLHF pipelines where some capability axes go
unmeasured at some checkpoints. The naive fix - impute a single global
prior for unmeasured axes - silently *rewards* candidates for avoiding
measurement of inherited weaknesses, because the prior usually exceeds a
truthfully-earned low score. The structural fix is lineage imputation:
an unmeasured axis should inherit the strongest prior evidence available
about it, which for an offspring is its parent's measured value. Dodging
then ties instead of winning; genuine improvement still wins. We found no
prior art naming this failure mode and would point to it if it exists.

With those four sealed, Sweep F produced a finding we trust for its *functional
form* rather than any fitness delta: **needle-env pass rate tracks context-
visibility almost exactly.** The gene `working_memory_size` controls how many
haystack lines enter the prompt, so visibility is computable: 8/38 = 21% at the
baseline, 16/38 = 42% after mutation. Observed pass rates: **1/4 (25%)** at wm=8;
**5/12 (42%) pooled across all three wm=16 candidates.** The mechanism is checkable
from raw episode counts - no fitness machinery involved. What we do *not* claim:
the single paired parent-child comparison (1/4 -> 2/4, n=4) is itself significant,
and one audit catch remains open: an earlier wiring bug meant `--group-size` was
silently ignored (all sweeps ran n=4), so these counts are exact but smaller than
the flags implied.

Two reproducible negatives also held across independent sweeps:
- **Unfaithful crossovers are reliably harmful**: −0.19…−0.31 across three sweeps — same band, three chances to be luck, three refusals.
- **No structural addition beat the plain baseline within the explored neighborhoods.** Scope: this budget (~100–250 calls/sweep), this 8B model, these five environments, and — before fix #3 — only the 1–2 genes the searcher actually visited. This is *not* evidence that structural search can't work; it's evidence that a search with blind targeting will burn its entire budget re-testing one gene.

AI-Fold asks a different question than ordinary agent frameworks:

> *What AI system should exist, and how can we experimentally discover it?*

It does this in two layers:

1. **Discovery Engine** (`aifold_discovery/`) — evolves *candidate AI systems* (agent architectures: planners, memory, verifiers, tools, critics) through real experiments on live LLMs, using multi-dimensional fitness and evolutionary search. **Atropos** is absorbed as its laboratory substrate (environments, rollouts, rewards).
2. **Neural Substrate** (`src/aifold/`) — an AlphaFold 3-derived model that represents AI systems as typed relational graphs (entities + relations), refines them with recycling, and generates *distributions over future trajectories* via latent diffusion with confidence ranking.

The strict boundary: **RL optimizes candidates; AI-Fold evolves candidates.**

---

### Phenotype & wiring audit (the silent-no-op class)

The `--group-size` catch was a *different bug category* than the four
measurement bugs: plain wiring that ran fine while doing something other than
what was asked. That class doesn't announce itself, so before calling this
locked we traced every knob to its consumption site (`tools/phenotype_audit.py`,
rerunnable):

| Knob | Consumed at | Status |
|---|---|---|
| `--generations / --n-envs / --max-calls / --difficulty / --pure-baseline` | runner loop, registry override, seed count | OK |
| `--group-size` | registry → spec → **env instance attribute** | fixed (was silently dropped) |
| `--max-concurrency` | backend semaphore | OK |
| genes: verifier, episodic/semantic memory, working_memory, decomposition, beam search, critic, retries, router, code tool | scaffolding call paths | OK |
| gene: browser tool | — | **was unwired** → mode=`unwired`, excluded from sampling until a runtime exists |
| mutation: raise_tool_budget | sandbox loop only | now self-gates on code-tool presence |

Consumption-site tracing alone missed two more of this class: `search_depth` was read by nothing (so `deepen_search` mutated a dead field), and `upgrade_search`'s first ladder step (none->bfs at depth 1) changed no behavior because refinement required depth > 1. Both were caught by `tools/behavior_audit.py`, which goes further than call-graph tracing: for every sampleable gene it runs paired OFF/ON scaffolds against a scripted backend and asserts the documented *behavioral* difference (verify stage fires, sandbox executes, context doubles, retry recovers). Current status: **12/12 behavioral checks pass**, covering all 11 live-sampleable genes. Rerun after any scaffold change; a gene whose check starts failing is a gene that stopped meaning anything.

Result: 11 live-sampleable mutations, each with verified consumption AND a passing behavioral difference check.
`unwired` joins `rl` in the excluded set - the phenotype contract now has three
states (changes behavior / needs weights / needs a runtime), and nothing lands
in the sampleable pool without proof it executes.

### Infrastructure reliability (separate from the science)

| Claim | Evidence |
|---|---|
| Survives free-tier rate limits | retry w/ exponential backoff + jitter rode out a 64-consecutive-429 storm |
| Scales to real sweeps | concurrent episodes + candidate parallelism; 234-call sweep in ~7 min |
| Deterministic scoring | all four environments verified locally (exact-match, executed unit tests, needle lookup) — no judge model in the measurement path |

---

## Live Quickstart

```bash
pip install torch numpy tqdm

# Configure any OpenAI-compatible backend (NVIDIA NIM shown)
cat > .env <<'EOF'
AIFOLD_BASE_URL=https://integrate.api.nvidia.com/v1
AIFOLD_API_KEY=nvapi-...
AIFOLD_MODEL=meta/llama-3.1-8b-instruct
EOF

# Decisive experiment: no hand-designed seeds, only live-active mutations,
# RL-side mutations quarantined until a weight trainer is wired.
python run_live_discovery.py --generations 4 --pop-size 10 --group-size 6 \
    --n-envs 3 --difficulty hard --pure-baseline --max-calls 4500
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

Every experiment persists full provenance to `aifold_runs/experiments/all_records.jsonl`: genome snapshot → environment → trajectory group → per-axis fitness delta → **vs-parent delta** (child vs strongest parent, computed on common measured axes so partial evaluations never bias the comparison) → failure diagnosis → mutation lineage.

Ranking uses a coverage-safe score (`ranking_fitness`): unmeasured axes are imputed at the parent's measured value, so an offspring that dodges measuring an inherited weakness ties the parent - never beats it. Reporting uses the plain measured composite; ranking uses coverage. Conflating the two was bug #4. Mutation sampling also enforces anti-repeat (a parent's own mutation is never resampled) and targeted-fallback (singleton bottleneck pools cannot dominate draws).

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

- ✅ Live loop closed end-to-end across 5 sweeps (~1,100 calls, 0 unhandled errors)
- ✅ Metric-integrity trail published: one false positive caught & retracted, one metric bias found & fixed before publication
- ⚠️ All live findings are directional (n≈1–3 lineages per mutation) — powered study spec in the header note
- ⚠️ Baseline composite quantizes coarsely at current episode counts (4–6 episodes/axis); resolution limits detectable effect sizes
- ⚠️ Neural substrate verified mechanically (forward/backward/sampling), not yet trained to convergence on real trajectory corpora
- ⚠️ Weight-training hook requires GPU infrastructure (`live/trainer.py::GrpoHook`)

## License

Apache 2.0 (matching AlphaFold 3 and Atropos).
