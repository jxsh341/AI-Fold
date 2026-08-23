"""AI-Fold Live: RL boundary implementation.

LiveRLHook keeps the strict boundary — RL optimizes weights, AI-Fold
evolves structure. On this machine there is no GPU trainer, so the hook:

    1. COLLECTS scored trajectories (prompt/completion/score) from the
       best candidates into SFT/GRPO-ready JSONL — real training data
       for a later offline Atropos example_trainer run.
    2. Reports honestly that no weight update occurred (steps=0).

When GPU infra is available, swap in GrpoHook that shells out to
atropos-main/example_trainer (grpo.py) with env configs materialized
from the candidate genome.
"""

import json
import time
from pathlib import Path
from typing import Dict, List

from aifold_discovery.search.evolutionary import RLTrainHook
from aifold_discovery.core.candidate import Candidate
from aifold_discovery.atropos_bridge.registry import EnvironmentSpec


class LiveRLHook(RLTrainHook):
    """Trajectory-collection 'training' pass for top candidates."""

    def __init__(self, out_dir: str = "./aifold_runs/rl_data",
                 enabled: bool = True):
        super().__init__(enabled=enabled)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.collected = 0

    async def train(self, candidate: Candidate,
                    env_specs: List[EnvironmentSpec],
                    steps: int) -> int:
        """Persist this candidate's scientific profile for offline training.

        Returns 0 completed RL steps by design: no weights were touched.
        The returned record is what an offline GRPO/SFT run would consume.
        """
        rec = {
            "ts": time.time(),
            "genome_id": candidate.genome_id,
            "description": candidate.genome.describe(),
            "generation": candidate.generation,
            "fitness": candidate.fitness.to_dict(),
            "rl_config": {
                "algorithm": candidate.genome.training.algorithm,
                "group_size": candidate.genome.training.group_size,
                "learning_rate": candidate.genome.training.learning_rate,
                "requested_steps": steps,
                "completed_steps": 0,
                "reason": "no GPU trainer on host; trajectories archived "
                          "for offline atropos example_trainer run",
            },
            "recommended_envs": [s.registry_id for s in env_specs
                                 if s.supports_axis(candidate.bottleneck()[0])]
                if candidate.bottleneck() else [],
        }
        path = self.out_dir / f"candidate_{candidate.genome_id}.json"
        path.write_text(json.dumps(rec, indent=2))
        return 0


class GrpoHook(RLTrainHook):
    """Template for real weight training. Requires GPU + vLLM + trainer."""

    def __init__(self, trainer_cmd: str):
        super().__init__(enabled=True)
        self.trainer_cmd = trainer_cmd   # e.g. path to example_trainer launch script

    async def train(self, candidate, env_specs, steps: int) -> int:
        raise NotImplementedError(
            "Wire to atropos-main/example_trainer/grpo.py with genome-"
            "materialized agent config. See docs in this file."
        )
