"""AI-Fold Live: environment registry wired to real verifiable environments."""

from typing import Optional

from aifold_discovery.atropos_bridge.registry import (
    EnvironmentRegistry, EnvironmentSpec,
)
from aifold_discovery.live.environments import make_env


def build_live_registry(group_size: int = 4,
                        only_ids: Optional[list] = None) -> EnvironmentRegistry:
    """Registry whose env_factory returns real live environments.

    Every spec keeps its typed experimental-genome metadata; difficulty
    maps straight into task generation parameters.
    """
    reg = EnvironmentRegistry()
    specs = [
        EnvironmentSpec(
            registry_id="math.reasoning.v4", family="math",
            capability_axes=["reasoning"], difficulty="medium",
            description="Generated multi-step word problems, exact-match scoring",
            reward_space="binary_correctness",
            evaluation_metrics=["pass_rate"],
            weight=2.0, group_size=group_size, source="aifold-live",
        ),
        EnvironmentSpec(
            registry_id="coding.execution.v3", family="coding",
            capability_axes=["coding"], difficulty="medium",
            description="Function synthesis verified by executed unit tests",
            reward_space="binary_correctness",
            constraints={"sandbox": "subprocess", "timeout_s": 10},
            weight=1.5, group_size=group_size, source="aifold-live",
        ),
        EnvironmentSpec(
            registry_id="agent.selfcorrection.v2", family="agents",
            capability_axes=["self_correction", "reasoning"], difficulty="medium",
            description="Trap problems; scores detect-and-recover behavior",
            reward_space="graded",
            evaluation_metrics=["recovery_rate", "pass_rate"],
            weight=1.75, group_size=group_size, source="aifold-live",
        ),
        EnvironmentSpec(
            registry_id="memory.long_context.v2", family="agents",
            capability_axes=["memory", "reasoning"], difficulty="hard",
            description=("Needle-in-haystack where working_memory_size gene "
                         "controls context assembly - structural effect on score"),
            reward_space="binary_correctness",
            weight=1.25, group_size=group_size, source="aifold-live",
        ),
        EnvironmentSpec(
            registry_id="generalization.heldout.v1", family="research",
            capability_axes=["generalization", "robustness"], difficulty="hard",
            description=("Hard-parameter variants of math/code tasks never "
                         "used for selection signal elsewhere"),
            reward_space="binary_correctness",
            weight=1.0, group_size=group_size, source="aifold-live",
        ),
    ]
    # generalization env reuses math generator at hard scale via subclassing-lite:
    from aifold_discovery.live.environments import LiveMathEnv, LiveCodeEnv

    class HeldOutMath(LiveMathEnv):
        name = "generalization.heldout.v1"
        capability_axes = ["generalization"]
        difficulty = "hard"

    class HeldOutCode(LiveCodeEnv):
        name = "generalization.heldout.v1"
        capability_axes = ["robustness"]
        difficulty = "hard"

    import aifold_discovery.live.environments as envs_mod
    envs_mod.ENV_CLASSES["generalization.heldout.v1"] = HeldOutMath

    if only_ids:
        specs = [s for s in specs if s.registry_id in only_ids]
    for s in specs:
        s.env_factory = (lambda rid=s.registry_id: make_env(rid))
        reg.register(s)
    return reg
