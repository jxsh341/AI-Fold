"""AI-Fold x Atropos: Environment Registry.

Turns Atropos environments into a first-class registry — the "experimental
genome" of AI-Fold. Each registered environment exposes:

    observation, action_space, trajectory, reward,
    constraints, difficulty, evaluation_metrics

and declares which *capability axis* it measures, so that trajectory
evidence collected from it attributes directly onto a FitnessVector.

The registry is deliberately decoupled from the Atropos rollout server:
an entry can point at (a) a real Atropos BaseEnv class, or (b) a local
mock adapter used for dry-run / CI evolution loops.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Type, Any


@dataclass
class EnvironmentSpec:
    """Registry entry describing one experimental environment."""

    registry_id: str                     # e.g. "math.reasoning.v4"
    family: str                          # math | coding | agents | research | tools | games | multimodal
    capability_axes: List[str]           # primary axes this env measures
    difficulty: str = "medium"           # easy | medium | hard
    description: str = ""

    # The actual environment implementation.
    # Either an Atropos BaseEnv subclass OR an AI-Fold MockEnv factory.
    env_factory: Optional[Callable[..., Any]] = None

    # Interface contract (the "experimental genome" fields)
    observation_space: str = "text"          # text | structured | multimodal
    action_space: str = "chat_completion"    # chat_completion | tool_calls | code_edit
    reward_space: str = "binary_correctness" # binary_correctness | graded | preference | multi_objective
    constraints: Dict[str, Any] = field(default_factory=dict)   # max_tokens, timeouts...
    evaluation_metrics: List[str] = field(default_factory=lambda: ["pass_rate"])

    # Sampling policy inside the evolution loop
    weight: float = 1.0                    # relative sampling weight
    min_batch_allocation: Optional[float] = None   # Atropos passthrough
    group_size: int = 16                   # Atropos passthrough

    # Provenance
    version: str = "v1"
    source: str = "aifold"                 # "atropos" | "aifold" | "community"

    def supports_axis(self, axis: str) -> bool:
        return axis in self.capability_axes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "registry_id": self.registry_id,
            "family": self.family,
            "capability_axes": list(self.capability_axes),
            "difficulty": self.difficulty,
            "description": self.description,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "reward_space": self.reward_space,
            "constraints": dict(self.constraints),
            "evaluation_metrics": list(self.evaluation_metrics),
            "weight": self.weight,
            "group_size": self.group_size,
            "version": self.version,
            "source": self.source,
            "has_env_impl": self.env_factory is not None,
        }


class EnvironmentRegistry:
    """Versioned registry of experimental environments."""

    def __init__(self):
        self._envs: Dict[str, EnvironmentSpec] = {}

    # ------------------------------------------------------------------
    def register(self, spec: EnvironmentSpec) -> None:
        if spec.registry_id in self._envs:
            raise ValueError(f"duplicate registry id: {spec.registry_id}")
        if not spec.capability_axes:
            raise ValueError(f"{spec.registry_id}: must declare >=1 capability axis")
        self._envs[spec.registry_id] = spec

    def get(self, registry_id: str) -> EnvironmentSpec:
        return self._envs[registry_id]

    def all(self) -> List[EnvironmentSpec]:
        return list(self._envs.values())

    def ids(self) -> List[str]:
        return sorted(self._envs.keys())

    def by_family(self, family: str) -> List[EnvironmentSpec]:
        return [e for e in self._envs.values() if e.family == family]

    def for_axis(self, axis: str) -> List[EnvironmentSpec]:
        """Environments that can measure a given capability axis."""
        return [e for e in self._envs.values() if e.supports_axis(axis)]

    def weakest_coverage(self, measured_counts: Dict[str, int]) -> Optional[EnvironmentSpec]:
        """Pick the env whose best-covered axis has least evidence so far.

        Drives evidence-balanced experimentation: prefer measuring what we
        know least about.
        """
        candidates = []
        for e in self._envs.values():
            min_count = min(
                measured_counts.get(ax, 0) for ax in e.capability_axes
            )
            candidates.append((-min_count * e.weight, e))
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1]

    # ------------------------------------------------------------------
    @classmethod
    def default(cls) -> "EnvironmentRegistry":
        """Seed registry mirroring the Atropos environment families."""
        reg = cls()
        seeds = [
            EnvironmentSpec(
                registry_id="math.reasoning.v4", family="math",
                capability_axes=["reasoning"], difficulty="medium",
                description="Multi-step mathematical reasoning with verifiable answers",
                reward_space="binary_correctness",
                constraints={"max_token_length": 16384},
                evaluation_metrics=["pass_rate", "pass_at_groupsize"],
                weight=2.0, source="atropos",
            ),
            EnvironmentSpec(
                registry_id="coding.swe.v7", family="coding",
                capability_axes=["coding", "planning"], difficulty="hard",
                description="Repository-level issue resolution (SWE-bench style)",
                reward_space="binary_correctness",
                constraints={"max_token_length": 32768, "sandbox": True},
                evaluation_metrics=["pass_rate", "resolved_rate"],
                weight=1.5, source="atropos",
            ),
            EnvironmentSpec(
                registry_id="coding.execution.v3", family="coding",
                capability_axes=["coding"], difficulty="medium",
                description="Code generation validated by execution tests",
                reward_space="graded",
                constraints={"max_token_length": 8192, "timeout_s": 30},
                weight=1.5, source="atropos",
            ),
            EnvironmentSpec(
                registry_id="agent.tooluse.v3", family="tools",
                capability_axes=["tool_use", "planning"], difficulty="medium",
                description="Multi-turn tool calling against real tool APIs",
                reward_space="graded",
                constraints={"max_turns": 10},
                evaluation_metrics=["pass_rate", "tool_efficiency"],
                weight=2.0, source="atropos",
            ),
            EnvironmentSpec(
                registry_id="memory.long_context.v2", family="agents",
                capability_axes=["memory", "reasoning"], difficulty="hard",
                description="Long-context retention and retrieval tasks",
                reward_space="graded",
                constraints={"context_tokens": 65536},
                weight=1.0, source="aifold",
            ),
            EnvironmentSpec(
                registry_id="agent.selfcorrection.v2", family="agents",
                capability_axes=["self_correction", "reasoning"], difficulty="medium",
                description="Detect-and-fix loops: initial attempt then verification+retry",
                reward_space="graded",
                constraints={"max_retries": 3},
                evaluation_metrics=["recovery_rate", "pass_rate"],
                weight=1.5, source="aifold",
            ),
            EnvironmentSpec(
                registry_id="research.web.v5", family="research",
                capability_axes=["planning", "tool_use"], difficulty="hard",
                description="Open-web research and synthesis tasks",
                reward_space="preference",
                constraints={"max_browse_steps": 20},
                weight=0.75, source="atropos",
            ),
            EnvironmentSpec(
                registry_id="games.multistep.v2", family="games",
                capability_axes=["planning"], difficulty="medium",
                description="Adversarial/multi-step games requiring lookahead",
                reward_space="binary_correctness",
                weight=0.75, source="atropos",
            ),
            EnvironmentSpec(
                registry_id="generalization.heldout.v1", family="research",
                capability_axes=["generalization", "robustness"], difficulty="hard",
                description="Held-out task suite never seen during training",
                reward_space="graded",
                weight=1.25, source="aifold",
            ),
        ]
        for s in seeds:
            reg.register(s)
        return reg

    def to_dict(self) -> Dict[str, Any]:
        return {rid: spec.to_dict() for rid, spec in self._envs.items()}
