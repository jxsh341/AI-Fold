"""AI-Fold Discovery: Selection + Novelty.

Selection operates on FitnessVectors (multi-dimensional), not scalar reward:
  - Pareto-front preservation keeps capability specialists alive
  - Composite ranking with uncertainty bonus explores under-measured candidates
  - Novelty search in fitness-space prevents premature convergence
"""

import math
import random
from typing import List, Optional, Tuple

from aifold_discovery.core.candidate import Candidate, Population
from aifold_discovery.core.fitness import FitnessVector


class Selector:
    def __init__(self, rng: Optional[random.Random] = None,
                 elite_frac: float = 0.25,
                 novelty_weight: float = 0.3,
                 min_samples: int = 4):
        self.rng = rng or random.Random()
        self.elite_frac = elite_frac
        self.novelty_weight = novelty_weight
        self.min_samples = min_samples   # composite confidence gate

    # ------------------------------------------------------------------
    @staticmethod
    def _uncertainty_bonus(c: Candidate) -> float:
        """Under-measured candidates get an exploration bonus."""
        total_n = sum(c.fitness.sample_counts.values())
        return 1.0 / (1.0 + math.log1p(total_n))

    def _novelty(self, c: Candidate, pop: List[Candidate]) -> float:
        """Mean distance to k nearest evaluated neighbors in fitness space."""
        others = [o for o in pop if o is not c and o.status == "evaluated"]
        if not others:
            return 1.0
        dists = sorted(c.fitness.distance(o.fitness) for o in others)
        k = min(3, len(dists))
        d = sum(dists[:k]) / k
        if math.isinf(d):
            return 1.5
        return min(1.0, d)

    def score(self, c: Candidate, pop: List[Candidate]) -> float:
        comp = c.ranking_fitness()
        enough = sum(1 for ax in c.fitness.measured_axes()
                     if c.fitness.sample_counts.get(ax, 0) >= self.min_samples) > 0
        base = comp if (comp is not None and enough) else -0.05
        return base + self.novelty_weight * (
            self._novelty(c, pop) + self._uncertainty_bonus(c)
        )

    # ------------------------------------------------------------------
    def rank(self, pop: Population) -> List[Tuple[float, Candidate]]:
        scored = [(self.score(c, pop.all()), c) for c in pop.all()]
        scored.sort(key=lambda t: -t[0])
        return scored

    def select_parents(self, pop: Population, n_pairs: int
                       ) -> List[Tuple[Candidate, Optional[Candidate]]]:
        """Tournament selection over the AI-Fold score; pairs for crossover."""
        ranked = [c for _s, c in self.rank(pop)]
        pool = ranked or []
        pairs = []

        def tournament(k: int = 3) -> Optional[Candidate]:
            if not pool:
                return None
            contenders = self.rng.sample(pool, min(k, len(pool)))
            return max(contenders, key=lambda c: self.score(c, pop.all()))

        for _ in range(n_pairs):
            a = tournament()
            b = tournament()
            pairs.append((a, b))
        return pairs

    def survivors(self, pop: Population, keep: int) -> List[Candidate]:
        """Elites by Pareto front first, then by composite score."""
        front = pop.pareto_front()
        rest = [c for c in pop.all() if c not in front]
        rest.sort(key=lambda c: -(self.score(c, pop.all())))
        out = front + rest
        return out[:keep]
