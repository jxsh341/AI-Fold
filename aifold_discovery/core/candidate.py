"""AI-Fold Discovery: Candidate and Population.

A Candidate couples a genome (structure) with its measured fitness
(capability profile) and full experimental lineage.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from aifold_discovery.core.genome import CandidateGenome
from aifold_discovery.core.fitness import FitnessVector, AXES


@dataclass
class Candidate:
    """An evolvable AI system: structure + measured capability + lineage."""

    genome: CandidateGenome
    fitness: FitnessVector = field(default_factory=FitnessVector)
    # Number of experiments this candidate has participated in.
    num_experiments: int = 0
    # Cumulative training steps applied via RL (Atropos-side).
    rl_steps_completed: int = 0
    # Status: "unevaluated" | "training" | "evaluated" | "retired"
    status: str = "unevaluated"
    # Composite fitness of the strongest parent at reproduction time.
    # Baseline for measuring whether THIS genome improved on what made it.
    parent_composite: Optional[float] = None
    # Full parent fitness snapshot â€” enables fair common-axes comparison
    # even while the child is only partially evaluated.
    parent_fitness: Optional[FitnessVector] = None

    @property
    def genome_id(self) -> str:
        return self.genome.genome_id

    @property
    def generation(self) -> int:
        return self.genome.generation

    def composite_fitness(self) -> Optional[float]:
        """Measured-axes composite (reporting). See also ranking_fitness()."""
        return self.fitness.composite()

    def ranking_fitness(self) -> float:
        """Coverage-safe score for SELECTION/RANKING.

        Unmeasured axes are imputed from the parent's measured value where
        available (an offspring that dodges measuring an inherited weakness
        ties, never wins), falling back to a weak global prior otherwise.
        Sweep F lesson: 0.835-without-memory-axis vs 0.543-with was a
        coverage artifact, not improvement.
        """
        priors = {}
        if self.parent_fitness is not None:
            for ax in AXES:
                pv = self.parent_fitness.get(ax)
                if pv is not None:
                    priors[ax] = pv
        cc = self.fitness.coverage_composite(priors=priors)
        return cc if cc is not None else float("-inf")

    def bottleneck(self):
        return self.fitness.weakest_axis()

    def to_dict(self) -> Dict:
        return {
            "genome": self.genome.to_dict(),
            "fitness": self.fitness.to_dict(),
            "num_experiments": self.num_experiments,
            "rl_steps_completed": self.rl_steps_completed,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Candidate":
        return cls(
            genome=CandidateGenome.from_dict(d["genome"]),
            fitness=FitnessVector.from_dict(d["fitness"]),
            num_experiments=d.get("num_experiments", 0),
            rl_steps_completed=d.get("rl_steps_completed", 0),
            status=d.get("status", "unevaluated"),
        )

    def describe(self) -> str:
        comp = self.composite_fitness()
        comp_str = f"{comp:.3f}" if comp is not None else "n/a"
        return (
            f"[{self.genome_id}] gen={self.genome.generation} "
            f"fitness={comp_str} exps={self.num_experiments} :: {self.genome.describe()}"
        )


class Population:
    """Managed set of candidates for one evolutionary run."""

    def __init__(self, max_size: int = 32):
        self.max_size = max_size
        self.candidates: Dict[str, Candidate] = {}

    def add(self, candidate: Candidate) -> None:
        self.candidates[candidate.genome_id] = candidate
        self._cull()

    def get(self, genome_id: str) -> Optional[Candidate]:
        return self.candidates.get(genome_id)

    def all(self) -> List[Candidate]:
        return list(self.candidates.values())

    def evaluated(self) -> List[Candidate]:
        return [c for c in self.all() if c.status == "evaluated"]

    def best(self, key=None) -> Optional[Candidate]:
        pool = self.evaluated() or self.all()
        if not pool:
            return None
        key = key or (lambda c: c.composite_fitness() if c.composite_fitness() is not None else float("-inf"))
        return max(pool, key=key)

    def pareto_front(self) -> List[Candidate]:
        """Candidates not dominated by any other evaluated candidate."""
        pool = self.evaluated()
        front = []
        for c in pool:
            dominated = False
            for other in pool:
                if other is c:
                    continue
                if other.fitness.dominates(c.fitness):
                    dominated = True
                    break
                # equal vectors: keep older/lower-id deterministically
                if not other.fitness.dominates(c.fitness) and other.fitness.distance(c.fitness) == 0 \
                        and other.genome_id < c.genome_id and other.fitness.dominates(c.fitness) is False:
                    pass  # tie; both stay unless strict domination exists elsewhere
            if not dominated:
                front.append(c)
        return front

    def _cull(self) -> None:
        """Retire weakest candidates when over capacity."""
        while len(self.candidates) > self.max_size:
            ranked = sorted(
                self.candidates.values(),
                key=lambda c: (
                    -(c.ranking_fitness()),
                    -c.num_experiments,
                ),
            )
            victim = ranked[-1]
            del self.candidates[victim.genome_id]

    def to_dict(self) -> Dict:
        return {"candidates": [c.to_dict() for c in self.all()]}

    @classmethod
    def from_dict(cls, d: Dict) -> "Population":
        pop = cls()
        for cd in d.get("candidates", []):
            pop.add(Candidate.from_dict(cd))
        return pop
