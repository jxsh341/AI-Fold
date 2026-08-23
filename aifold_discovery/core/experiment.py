"""AI-Fold Discovery: Experiment Records.

Atropos trajectories become scientific evidence. An ExperimentRecord is the
atomic unit of that evidence:

    candidate genome  ->  environment  ->  trajectory group  ->  outcome
                                    \-> failure analysis -> next mutation

Unlike plain RL (prompt -> model -> response -> reward -> discard), AI-Fold
persists the full provenance so the system builds a scientific history of
what works, for which capability, and why.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import uuid


@dataclass
class TrajectoryEvidence:
    """One scored trajectory group from an Atropos environment.

    Mirrors atroposlib.envs.base.ScoredDataGroup but adds AI-Fold metadata:
    per-item scores plus derived diagnostics used for fitness attribution.
    """

    env_name: str
    # Raw Atropos fields (kept verbatim so nothing is lost)
    tokens: Optional[List[List[int]]] = None
    masks: Optional[List[List[int]]] = None
    scores: Optional[List[float]] = None
    advantages: Optional[Any] = None
    messages: Optional[Any] = None
    # Derived evidence fields
    n_items: int = 0
    mean_score: float = 0.0
    pass_rate: float = 0.0            # fraction of items with score > 0
    score_variance: float = 0.0       # learning-signal richness in the group
    mean_length_tokens: float = 0.0   # efficiency proxy
    had_failures: bool = False        # any item scored <= 0
    had_self_correction: bool = False  # retry/recovery observed (env-tagged)
    tool_call_count: int = 0          # tool-use intensity (env-tagged)
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_atropos_group(
        cls,
        group: Dict[str, Any],
        env_name: str,
        tags: Optional[Dict[str, Any]] = None,
    ) -> "TrajectoryEvidence":
        """Build evidence from a raw Atropos ScoredDataGroup dict."""
        scores = list(group.get("scores") or [])
        masks = list(group.get("masks") or [])
        n = len(scores)
        mean = sum(scores) / n if n else 0.0
        var = sum((s - mean) ** 2 for s in scores) / n if n else 0.0
        lengths = [sum(1 for m in msk if m != -100) for msk in masks] if masks else []
        tags = tags or {}
        return cls(
            env_name=env_name,
            tokens=group.get("tokens"),
            masks=masks,
            scores=scores,
            advantages=group.get("advantages"),
            messages=group.get("messages"),
            n_items=n,
            mean_score=mean,
            pass_rate=(sum(1 for s in scores if s > 0) / n) if n else 0.0,
            score_variance=var,
            mean_length_tokens=(sum(lengths) / len(lengths)) if lengths else 0.0,
            had_failures=any(s <= 0 for s in scores),
            had_self_correction=bool(tags.get("self_corrected", False)),
            tool_call_count=int(tags.get("tool_calls", 0)),
            extras=dict(group.get("group_overrides") or {}),
        )


@dataclass
class ExperimentRecord:
    """Full provenance of one candidate-in-environment experiment."""

    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    genome_id: str = ""
    genome_snapshot: Dict[str, Any] = field(default_factory=dict)
    generation: int = 0

    env_registry_id: str = ""          # e.g. "coding.swe.v7"
    env_capability_axis: str = ""      # primary axis this environment measures
    difficulty: str = "medium"

    # Training phase outcome (RL-side, via Atropos trainer)
    rl_steps_run: int = 0
    pre_train_reward: Optional[float] = None

    # Evaluation phase outcome (evidence-based)
    evidences: List[TrajectoryEvidence] = field(default_factory=list)

    # Post-evaluation analysis
    fitness_delta: Dict[str, float] = field(default_factory=dict)
    failure_summary: str = ""
    diagnosis: str = ""                # e.g. "weak dependency reasoning"

    mutation_applied: str = ""         # mutation that produced this candidate
    parent_ids: List[str] = field(default_factory=list)

    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    # ------------------------------------------------------------------
    def add_evidence(self, ev: TrajectoryEvidence) -> None:
        self.evidences.append(ev)

    def aggregate_scores(self) -> Dict[str, float]:
        """Aggregate evidence into {axis: score} contributions."""
        agg: Dict[str, List[float]] = {}
        for ev in self.evidences:
            axis = self.env_capability_axis or "reasoning"
            if ev.n_items == 0:
                continue
            agg.setdefault(axis, []).append(ev.pass_rate)
        return {
            ax: sum(v) / len(v) for ax, v in agg.items() if v
        }

    def finalize(self, fitness_delta: Dict[str, float], diagnosis: str = "") -> None:
        self.fitness_delta = fitness_delta
        self.diagnosis = diagnosis
        self.completed_at = time.time()

    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "genome_id": self.genome_id,
            "genome_snapshot": self.genome_snapshot,
            "generation": self.generation,
            "env_registry_id": self.env_registry_id,
            "env_capability_axis": self.env_capability_axis,
            "difficulty": self.difficulty,
            "rl_steps_run": self.rl_steps_run,
            "pre_train_reward": self.pre_train_reward,
            "evidences": [
                {
                    "env_name": e.env_name,
                    "n_items": e.n_items,
                    "mean_score": e.mean_score,
                    "pass_rate": e.pass_rate,
                    "score_variance": e.score_variance,
                    "mean_length_tokens": e.mean_length_tokens,
                    "had_failures": e.had_failures,
                    "had_self_correction": e.had_self_correction,
                    "tool_call_count": e.tool_call_count,
                }
                for e in self.evidences
            ],
            "fitness_delta": self.fitness_delta,
            "failure_summary": self.failure_summary,
            "diagnosis": self.diagnosis,
            "mutation_applied": self.mutation_applied,
            "parent_ids": list(self.parent_ids),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class ExperimentStore:
    """Persistent scientific history of all experiments (the lab notebook)."""

    def __init__(self):
        self.records: Dict[str, ExperimentRecord] = {}

    def add(self, rec: ExperimentRecord) -> None:
        self.records[rec.experiment_id] = rec

    def for_genome(self, genome_id: str) -> List[ExperimentRecord]:
        return [r for r in self.records.values() if r.genome_id == genome_id]

    def lineage(self, genome_id: str) -> List[ExperimentRecord]:
        """All experiments in the ancestral chain of a genome."""
        out: List[ExperimentRecord] = []
        frontier = [genome_id]
        seen = set()
        while frontier:
            gid = frontier.pop()
            if gid in seen:
                continue
            seen.add(gid)
            for r in self.for_genome(gid):
                out.append(r)
                frontier.extend(r.parent_ids)
        return out

    def discovery_stats(self) -> Dict[str, Any]:
        """What has the search learned so far?"""
        by_mutation: Dict[str, List[float]] = {}
        by_env: Dict[str, List[float]] = {}
        for r in self.records.values():
            if not r.evidences:
                continue
            key = r.mutation_applied or "origin"
            delta = sum(r.fitness_delta.values()) if r.fitness_delta else 0.0
            by_mutation.setdefault(key, []).append(delta)
            by_env.setdefault(r.env_registry_id, []).append(
                sum(e.pass_rate for e in r.evidences) / max(1, len(r.evidences))
            )
        avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
        return {
            "total_experiments": len(self.records),
            "avg_delta_by_mutation": {k: round(avg(v), 4) for k, v in by_mutation.items()},
            "avg_pass_rate_by_env": {k: round(avg(v), 4) for k, v in by_env.items()},
        }
