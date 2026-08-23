"""AI-Fold Discovery: Evolutionary Search Engine.

The core loop:

    for generation in 1..G:
        1. SELECT parents (Pareto + novelty + uncertainty)
        2. MUTATE/CROSSOVER into children (fitness-targeted hypotheses)
        3. SCHEDULE experiments: child x environment (Atropos substrate)
        4. COLLECT trajectory evidence -> attribute to FitnessVector
        5. DIAGNOSE regressions, record scientific history
        6. RL-train promising candidates via Atropos trainer (hook)
        7. CULL and carry population forward

RL optimizes candidates. AI-Fold evolves candidates. That boundary is
preserved strictly: this module never touches weights.
"""

import asyncio
import random
import time
from typing import Callable, Dict, List, Optional

from aifold_discovery.core.genome import CandidateGenome, baseline_genome
from aifold_discovery.core.candidate import Candidate, Population
from aifold_discovery.core.fitness import FitnessVector
from aifold_discovery.core.experiment import (
    ExperimentRecord, ExperimentStore, TrajectoryEvidence,
)
from aifold_discovery.atropos_bridge.registry import (
    EnvironmentRegistry, EnvironmentSpec,
)
from aifold_discovery.atropos_bridge.adapter import run_environment
from aifold_discovery.evaluation.fitness import update_fitness, diagnose
from aifold_discovery.evolution.mutation import Mutator, Crossover
from aifold_discovery.evolution.selection import Selector


class RLTrainHook:
    """Optional hook into an actual Atropos trainer.

    Implement `train(candidate, env_specs, steps)` to launch GRPO/SFT runs
    against real environments. In dry-run mode this hook is a no-op that
    just increments the step counter.
    """

    # True only when this hook actually updates weights. Governs whether
    # RL-side mutations may be sampled (they are inert otherwise).
    trains_weights: bool = False

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    async def train(self, candidate: Candidate,
                    env_specs: List[EnvironmentSpec],
                    steps: int) -> int:
        if not self.enabled:
            return 0
        # Real implementation would:
        #   - materialize candidate genome into agent scaffolding config
        #   - register envs with the Atropos rollout server
        #   - launch example_trainer/grpo.py against those envs
        #   - wait for completion, return steps completed
        raise NotImplementedError(
            "Wire your Atropos trainer here (example_trainer/grpo.py entrypoint)"
        )


class DiscoveryEngine:
    def __init__(
        self,
        registry: Optional[EnvironmentRegistry] = None,
        pop_size: int = 12,
        seed: int = 0,
        rl_hook: Optional[RLTrainHook] = None,
        groups_per_experiment: int = 2,
        difficulty_scale: float = 1.0,
        on_event: Optional[Callable[[str, Dict], None]] = None,
    ):
        self.registry = registry or EnvironmentRegistry.default()
        self.population = Population(max_size=pop_size)
        self.store = ExperimentStore()
        self.mutator = Mutator(rng=random.Random(seed))
        self.crossover = Crossover(rng=random.Random(seed))
        self.selector = Selector(rng=random.Random(seed))
        self.rl_hook = rl_hook or RLTrainHook(enabled=False)
        self.rng = random.Random(seed)
        self.groups_per_experiment = groups_per_experiment
        self.difficulty_scale = difficulty_scale
        self.on_event = on_event or (lambda kind, payload: None)

    # ------------------------------------------------------------------
    def _emit(self, kind: str, payload: Dict):
        self.on_event(kind, payload)

    def seed_population(self, n_variants: int = 3) -> List[Candidate]:
        """Generation 0: baseline plus a couple of cheap hand-picked variants."""
        seeds = [Candidate(genome=baseline_genome())]

        v1 = baseline_genome().clone()
        from aifold_discovery.core.genome import ModelComponent
        v1.model.verifier_enabled = True
        v1.mutation_history.append("seed:add_verifier")
        seeds.append(Candidate(genome=v1))

        if n_variants > 2:
            v2 = baseline_genome().clone()
            from aifold_discovery.core.genome import ToolComponent
            v2.tools.enabled_tools = ["code"]
            v2.mutation_history.append("seed:add_tool_code")
            seeds.append(Candidate(genome=v2))

        for s in seeds[:n_variants]:
            self.population.add(s)
        self._emit("population_seeded", {"size": len(self.population.all())})
        return self.population.all()

    # ------------------------------------------------------------------
    async def _run_single_experiment(self, candidate: Candidate,
                                     spec: EnvironmentSpec) -> ExperimentRecord:
        pre_fitness = FitnessVector.from_dict(candidate.fitness.to_dict())

        rec = ExperimentRecord(
            genome_id=candidate.genome_id,
            genome_snapshot=candidate.genome.canonical_dict(),
            generation=candidate.generation,
            env_registry_id=spec.registry_id,
            env_capability_axis=",".join(spec.capability_axes),
            difficulty=spec.difficulty,
            mutation_applied=(candidate.genome.mutation_history[-1]
                              if candidate.genome.mutation_history else "origin"),
            parent_ids=list(candidate.genome.parent_ids),
        )

        evidences = await run_environment(
            spec, candidate.genome,
            n_groups=self.groups_per_experiment,
            seed=hash((candidate.genome_id, spec.registry_id)) % (2**31),
        )
        for ev in evidences:
            rec.add_evidence(ev)

        # Attribute evidence onto the fitness vector
        candidate.fitness = update_fitness(candidate.fitness, rec)
        delta = {}
        for ax in set(pre_fitness.scores) | set(candidate.fitness.scores):
            a, b = pre_fitness.get(ax), candidate.fitness.get(ax)
            if b is not None:
                delta[ax] = round(b - (a if a is not None else b), 4)

        # Mutation effectiveness: child vs strongest parent, compared on
        # COMMON measured axes only (fair while child is mid-evaluation).
        vs_parent = None
        if candidate.parent_fitness is not None:
            common = [ax for ax in candidate.fitness.measured_axes()
                      if candidate.parent_fitness.get(ax) is not None]
            if common:
                c = sum(candidate.fitness.get(ax) for ax in common) / len(common)
                p = sum(candidate.parent_fitness.get(ax) for ax in common) / len(common)
                vs_parent = round(c - p, 4)

        diagnosis = diagnose(rec, pre_fitness, candidate.fitness)
        rec.finalize(fitness_delta=delta, diagnosis=diagnosis,
                     vs_parent_delta=vs_parent)

        candidate.num_experiments += 1
        candidate.status = "evaluated"
        self.store.add(rec)

        self._emit("experiment_done", {
            "genome": candidate.genome_id,
            "env": spec.registry_id,
            "delta": delta,
            "vs_parent": vs_parent,
            "diagnosis": diagnosis,
        })
        return rec

    # ------------------------------------------------------------------
    async def evaluate_candidate(self, candidate: Candidate,
                                 n_envs: int = 2,
                                 prefer_weakest_coverage: bool = True) -> List[ExperimentRecord]:
        """Run one candidate through evidence-balanced environments."""
        records = []
        measured_counts: Dict[str, int] = dict(candidate.fitness.sample_counts)
        tried: set = set()
        for _ in range(n_envs):
            spec = (self.registry.weakest_coverage(measured_counts)
                    if prefer_weakest_coverage else None)
            if spec is None or spec.registry_id in tried:
                pool = [s for s in self.registry.all() if s.registry_id not in tried]
                if not pool:
                    break
                weights = [s.weight for s in pool]
                spec = self.rng.choices(pool, weights=weights, k=1)[0]
            tried.add(spec.registry_id)
            rec = await self._run_single_experiment(candidate, spec)
            records.append(rec)
            for ax in spec.capability_axes:
                measured_counts[ax] = measured_counts.get(ax, 0) + sum(
                    e.n_items for e in rec.evidences)
        return records

    # ------------------------------------------------------------------
    def next_generation_candidates(self, n_children: int) -> List[Candidate]:
        """Selection -> mutation/crossover -> children with reset measurements."""
        pairs = self.selector.select_parents(self.population, n_pairs=n_children)
        children: List[Candidate] = []
        for parent_a, parent_b in pairs:
            if parent_a is None:
                continue
            bottleneck = parent_a.bottleneck()
            axis = bottleneck[0] if bottleneck else None

            use_crossover = (parent_b is not None and parent_b is not parent_a
                             and self.rng.random() < 0.35)
            if use_crossover:
                genome, label = self.crossover.crossover(parent_a.genome, parent_b.genome)
            else:
                allow_rl = bool(getattr(self.rl_hook, "trains_weights", False))
                avoid = (parent_a.genome.mutation_history[-1]
                         if parent_a.genome.mutation_history else None)
                genome, label = self.mutator.mutate(
                    parent_a.genome,
                    bottleneck_axis=axis,
                    allow_rl=allow_rl,
                    avoid=avoid)

            child = Candidate(genome=genome)
            child.status = "unevaluated"
            # Record parent strength at conception for vs-parent deltas.
            parents = [p for p in (parent_a, parent_b) if p is not None]
            baselines = [p.composite_fitness() for p in parents
                         if p.composite_fitness() is not None]
            child.parent_composite = max(baselines) if baselines else None
            strongest = max(parents, key=lambda p: p.composite_fitness() or -1) \
                if baselines else None
            child.parent_fitness = (FitnessVector.from_dict(
                strongest.fitness.to_dict()) if strongest is not None else None)
            children.append(child)
            self._emit("child_created", {
                "child": genome.genome_id,
                "parents": list(genome.parent_ids),
                "mutation": label,
                "targeted_axis": axis,
            })
        return children

    # ------------------------------------------------------------------
    async def maybe_rl_train(self, candidate: Candidate,
                             top_fraction: float = 0.3) -> int:
        """Send top-slice candidates to the Atropos trainer (if wired)."""
        ranked = self.selector.rank(self.population)
        cutoff = max(1, int(len(ranked) * top_fraction))
        top_ids = {c.genome_id for _s, c in ranked[:cutoff]}
        if candidate.genome_id not in top_ids:
            return 0
        steps = await self.rl_hook.train(
            candidate, self.registry.all(),
            steps=candidate.genome.training.total_steps,
        )
        candidate.rl_steps_completed += steps
        return steps

    # ------------------------------------------------------------------
    async def run_generation(self, n_envs_per_candidate: int = 2) -> List[Candidate]:
        """One full evolutionary generation over the current population."""
        t0 = time.time()
        self._emit("generation_start", {
            "gen": max((c.generation for c in self.population.all()), default=0),
            "pop_size": len(self.population.all()),
        })

        # 1-2. Evaluate everyone not yet evaluated
        for cand in list(self.population.all()):
            if cand.status != "evaluated":
                await self.evaluate_candidate(cand, n_envs=n_envs_per_candidate)

        # 3. RL-train the top slice (no-op unless rl_hook.enabled)
        for cand in list(self.population.all()):
            await self.maybe_rl_train(cand)

        best = self.population.best()
        self._emit("generation_best", {
            "genome": best.genome_id if best else None,
            "fitness": best.composite_fitness() if best else None,
            "desc": best.genome.describe() if best else "",
        })

        # 4. Reproduce
        children = self.next_generation_candidates(
            n_children=max(2, len(self.population.all()) // 2))
        for ch in children:
            self.population.add(ch)

        self._emit("generation_end", {
            "duration_s": round(time.time() - t0, 2),
            "new_children": len(children),
        })
        return children

    # ------------------------------------------------------------------
    async def discover(self, generations: int = 3,
                       n_envs_per_candidate: int = 2) -> Dict:
        """Run the full discovery loop."""
        if not self.population.all():
            self.seed_population()

        for g in range(generations):
            await self.run_generation(n_envs_per_candidate=n_envs_per_candidate)

        best = self.population.best()
        summary = {
            "generations_run": generations,
            "best_genome": best.to_dict() if best else None,
            "pareto_front": [c.describe() for c in self.population.pareto_front()],
            "discovery_stats": self.store.discovery_stats(),
            "final_population": [c.describe() for c in self.population.all()],
        }
        self._emit("discovery_complete", summary)
        return summary
