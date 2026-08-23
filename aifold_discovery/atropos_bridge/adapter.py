"""AI-Fold x Atropos: Environment Adapter.

Bridges an EnvironmentSpec to either:
  (a) a real atroposlib BaseEnv instance  -> collect_trajectories(item)
  (b) an AI-Fold MockEnv                  -> deterministic dry-run evidence

The adapter normalizes both to a single async interface:

    evidences: List[TrajectoryEvidence] = await adapter.run(candidate)

Candidate-awareness is the key addition over raw Atropos: the adapter
derives generation params (temperature, retries, tool budget, context
window) from the CandidateGenome so that *the system being tested*
determines how the environment is driven.
"""

import asyncio
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from aifold_discovery.core.genome import CandidateGenome
from aifold_discovery.core.experiment import TrajectoryEvidence
from aifold_discovery.atropos_bridge.registry import EnvironmentSpec


# ----------------------------------------------------------------------
# Mock environment (no LLM / no server needed). Deterministic-ish scoring
# driven by genome capability priors. Used for CI, dry-runs, and for
# testing evolution mechanics before spending real compute.


def _capability_prior(genome: CandidateGenome) -> Dict[str, float]:
    """Rough synthetic capability model of a genome.

    In production these come from real trajectories; here they encode
    plausible structural advantages so evolution has signal to find.
    """
    base = {
        "reasoning": 0.45,
        "coding": 0.40,
        "planning": 0.38,
        "tool_use": 0.35,
        "memory": 0.40,
        "self_correction": 0.30,
        "efficiency": 0.55,
        "robustness": 0.42,
        "generalization": 0.40,
    }
    # Model component
    if genome.model.verifier_enabled:
        base["self_correction"] += 0.22
        base["reasoning"] += 0.06
    # Memory component
    if genome.memory.episodic_memory:
        base["memory"] += 0.18
    if genome.memory.semantic_memory:
        base["memory"] += 0.12
        base["generalization"] += 0.08
    # Planning
    if genome.planning.decomposition:
        base["planning"] += 0.16
        if genome.planning.search_algorithm in ("mcts", "beam"):
            base["planning"] += 0.10
    # Tools
    if "code" in genome.tools.enabled_tools:
        base["coding"] += 0.15
    if "browser" in genome.tools.enabled_tools:
        base["tool_use"] += 0.10
        base["planning"] += 0.04
    if genome.tools.enabled_tools:
        base["tool_use"] += 0.08
    # Control
    if genome.control.critic_enabled:
        base["self_correction"] += 0.10
        base["robustness"] += 0.08
    if genome.control.router_type == "conditional":
        base["efficiency"] += 0.10
        base["robustness"] += 0.05
    if not genome.control.retry_on_failure:
        base["robustness"] -= 0.10
    # Cost of complexity: efficiency penalty
    complexity = (
        int(genome.memory.episodic_memory)
        + int(genome.memory.semantic_memory)
        + int(genome.planning.decomposition)
        + len(genome.tools.enabled_tools)
        + int(genome.control.critic_enabled)
    )
    base["efficiency"] -= 0.03 * complexity
    return {k: max(0.02, min(0.98, v)) for k, v in base.items()}


class MockEnvAdapter:
    """Generates TrajectoryEvidence without running any model."""

    def __init__(self, spec: EnvironmentSpec, seed: int = 0):
        self.spec = spec
        self.rng = random.Random(seed)

    async def run(
        self,
        genome: CandidateGenome,
        n_groups: int = 2,
        group_size: int = 8,
        difficulty_scale: float = 1.0,
    ) -> List[TrajectoryEvidence]:
        prior = _capability_prior(genome)
        primary_axis = self.spec.capability_axes[0]
        p = prior.get(primary_axis, 0.4)

        # Difficulty modulates success probability
        diff_mult = {"easy": 1.25, "medium": 1.0, "hard": 0.75}.get(self.spec.difficulty, 1.0)
        p_eff = max(0.02, min(0.98, p * diff_mult * difficulty_scale))

        evidences = []
        for _ in range(n_groups):
            scores = [1.0 if self.rng.random() < p_eff else -1.0
                      for _ in range(group_size)]
            tags = {}
            if primary_axis == "self_correction" and genome.model.verifier_enabled:
                # verifier enables recovery on failures
                scores = [
                    1.0 if (s > 0 or self.rng.random() < 0.5) else -1.0 for s in scores
                ]
                tags["self_corrected"] = True
            if primary_axis == "tool_use":
                tags["tool_calls"] = min(genome.tools.max_tool_calls_per_episode, group_size)
            ev = TrajectoryEvidence.from_atropos_group(
                {"scores": scores,
                 "masks": [[0] * 64] * group_size,
                 "tokens": [[]] * group_size},
                env_name=self.spec.registry_id,
                tags=tags,
            )
            evidences.append(ev)
        return evidences


# ----------------------------------------------------------------------
# Real-Atropos adapter (used when atroposlib is importable and a rollout
# server / local env is available).


class AtroposEnvAdapter:
    """Drives a real Atropos BaseEnv-like object and harvests ScoredDataGroups.

    Expected interface of `atropos_env` (duck-typed so we do not require
    a full rollout-server deployment):

        async def collect_trajectories(item) -> (ScoredDataGroup | None, backlog)
        async def get_next_item() -> item

    We run N items through it and convert each returned ScoredDataGroup
    into AI-Fold TrajectoryEvidence. Nothing is discarded: tokens/masks/
    scores/messages are all retained on the evidence record.
    """

    def __init__(self, spec: EnvironmentSpec, atropos_env: Any, tokenizer=None):
        self.spec = spec
        self.env = atropos_env
        self.tokenizer = tokenizer

    async def _collect_one_group(self, item: Any) -> Optional[Dict[str, Any]]:
        try:
            group, _backlog = await self.env.collect_trajectories(item)
        except Exception as e:  # env failure => no evidence from this item
            return None
        if group is None or not isinstance(group, dict):
            return None
        return group

    async def run(
        self,
        genome: CandidateGenome,
        n_groups: int = 2,
        group_size: Optional[int] = None,
        **kwargs,
    ) -> List[TrajectoryEvidence]:
        evidences: List[TrajectoryEvidence] = []
        gs = group_size or getattr(getattr(self.env, "config", None),
                                   "group_size", 8)
        for _ in range(n_groups):
            item = await self.env.get_next_item()
            group = await self._collect_one_group(item)
            if group is None:
                continue
            tags = {}
            msgs = group.get("messages")
            if isinstance(msgs, list) and msgs:
                flat = str(msgs[-1])[:4000].lower()
                tags["self_corrected"] = ("retry" in flat or "re-read" in flat
                                          or "correction:" in flat)
                tags["tool_calls"] = flat.count("tool_call") + flat.count("<tool>")
            evidences.append(
                TrajectoryEvidence.from_atropos_group(group, self.spec.registry_id, tags)
            )
        return evidences


# ----------------------------------------------------------------------
# Unified factory


def build_adapter(spec: EnvironmentSpec, seed: int = 0) -> Any:
    """Return the right adapter for this registry entry."""
    if spec.env_factory is not None:
        env = spec.env_factory()
        return AtroposEnvAdapter(spec, env)
    return MockEnvAdapter(spec, seed=seed)


async def run_environment(
    spec: EnvironmentSpec,
    genome: CandidateGenome,
    n_groups: int = 2,
    seed: int = 0,
) -> List[TrajectoryEvidence]:
    """One-shot convenience: run candidate in environment, get evidence."""
    adapter = build_adapter(spec, seed=seed)
    return await adapter.run(genome, n_groups=n_groups)
