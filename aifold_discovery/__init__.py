"""AI-Fold Discovery Engine.

Absorbs Atropos as the experiment substrate (environments, rollouts,
rewards) and layers on top of it: candidate genomes, multi-dimensional
fitness, evolutionary search, and a scientific experiment archive.

    RL optimizes candidates.  AI-Fold evolves candidates.
"""

from aifold_discovery.core.genome import CandidateGenome, baseline_genome
from aifold_discovery.core.candidate import Candidate, Population
from aifold_discovery.core.fitness import FitnessVector, AXES
from aifold_discovery.core.experiment import (
    ExperimentRecord, ExperimentStore, TrajectoryEvidence,
)
from aifold_discovery.atropos_bridge.registry import (
    EnvironmentRegistry, EnvironmentSpec,
)

__all__ = [
    "CandidateGenome", "baseline_genome",
    "Candidate", "Population",
    "FitnessVector", "AXES",
    "ExperimentRecord", "ExperimentStore", "TrajectoryEvidence",
    "EnvironmentRegistry", "EnvironmentSpec",
]
