"""AI-Fold Discovery: Multi-dimensional Fitness.

A FitnessVector is a candidate's measured capability profile across
independent capability axes. Upgrade over scalar RL reward:

    reward  = 0.82                          (what plain RL gives you)
    fitness = {reasoning: 0.91, coding: 0.87, ...}   (what AI-Fold gives you)

Enables multi-objective evolutionary search, Pareto selection, and
per-axis bottleneck diagnosis of candidates.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

AXES = [
    "reasoning",
    "coding",
    "planning",
    "tool_use",
    "memory",
    "self_correction",
    "efficiency",
    "robustness",
    "generalization",
]


@dataclass
class FitnessVector:
    """Per-axis scores in [0, 1]. Missing measurements are None."""

    scores: Dict[str, Optional[float]] = field(default_factory=dict)
    sample_counts: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        for ax in AXES:
            self.scores.setdefault(ax, None)
            self.sample_counts.setdefault(ax, 0)

    def set(self, axis: str, value: float, n_samples: int = 1) -> None:
        if axis not in AXES:
            raise ValueError("unknown axis '%s'" % axis)
        self.scores[axis] = max(0.0, min(1.0, float(value)))
        self.sample_counts[axis] += int(n_samples)

    def get(self, axis: str) -> Optional[float]:
        return self.scores.get(axis)

    def measured_axes(self) -> List[str]:
        return [ax for ax in AXES if self.scores.get(ax) is not None]

    def composite(
        self,
        weights: Optional[Dict[str, float]] = None,
        min_confidence: int = 0,
    ) -> Optional[float]:
        """Weighted mean over MEASURED axes with enough evidence.

        Measurement/reporting metric. For ranking candidates with different
        axis coverage, use coverage_composite() instead.
        """
        w = weights or {}
        num, den = 0.0, 0.0
        for ax in self.measured_axes():
            if self.sample_counts.get(ax, 0) < min_confidence:
                continue
            weight = w.get(ax, 1.0)
            num += weight * self.scores[ax]
            den += weight
        return num / den if den > 0 else None

    def coverage_composite(self, prior: float = 0.35,
                           weights: Optional[Dict[str, float]] = None,
                           min_confidence: int = 0,
                           priors: Optional[Dict[str, float]] = None
                           ) -> Optional[float]:
        """Composite over ALL axes, imputing unmeasured ones.

        RANKING-SAFE comparison across different axis-coverage sets.
        `priors` allows per-axis imputation values; pass the PARENT's
        measured scores so an offspring that dodges measuring an inherited
        weakness inherits that weakness's value instead of a free upgrade.
        """
        w = weights or {}
        p = priors or {}
        num, den = 0.0, 0.0
        for ax in AXES:
            weight = w.get(ax, 1.0)
            v = self.scores.get(ax)
            if v is not None and self.sample_counts.get(ax, 0) >= min_confidence:
                num += weight * v
            else:
                num += weight * p.get(ax, prior)
            den += weight
        return num / den if den > 0 else None
        w = weights or {}
        num, den = 0.0, 0.0
        for ax in self.measured_axes():
            if self.sample_counts.get(ax, 0) < min_confidence:
                continue
            weight = w.get(ax, 1.0)
            num += weight * self.scores[ax]
            den += weight
        return num / den if den > 0 else None

    def merge(self, other: "FitnessVector", alpha: float = 0.5) -> "FitnessVector":
        """Running-mean merge per axis: new = alpha*other + (1-alpha)*self."""
        out = FitnessVector()
        for ax in AXES:
            a, b = self.scores.get(ax), other.scores.get(ax)
            na = self.sample_counts.get(ax, 0)
            nb = other.sample_counts.get(ax, 0)
            if a is None and b is None:
                continue
            elif b is None:
                out.scores[ax], out.sample_counts[ax] = a, na
            elif a is None:
                out.scores[ax], out.sample_counts[ax] = b, nb
            else:
                out.scores[ax] = alpha * b + (1 - alpha) * a
                out.sample_counts[ax] = na + nb
        return out

    def dominates(self, other: "FitnessVector") -> bool:
        """Pareto dominance on commonly measured axes (strictly better somewhere)."""
        compared = 0
        strictly_better = False
        for ax in set(self.measured_axes()) & set(other.measured_axes()):
            m, o = self.scores[ax], other.scores[ax]
            if m < o:
                return False
            if m > o:
                strictly_better = True
            compared += 1
        return compared > 0 and strictly_better

    def distance(self, other: "FitnessVector") -> float:
        """Euclidean distance over commonly measured axes (novelty metric)."""
        common = set(self.measured_axes()) & set(other.measured_axes())
        if not common:
            return math.inf
        s = sum((self.scores[a] - other.scores[a]) ** 2 for a in common)
        return math.sqrt(s)

    def weakest_axis(self) -> Optional[Tuple[str, float]]:
        """Axis with lowest score - the candidate's bottleneck."""
        best = None
        for ax in self.measured_axes():
            v = self.scores[ax]
            if best is None or v < best[1]:
                best = (ax, v)
        return best

    def to_dict(self) -> Dict:
        return {
            "scores": dict(self.scores),
            "sample_counts": dict(self.sample_counts),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "FitnessVector":
        fv = cls()
        fv.scores.update(d.get("scores", {}))
        fv.sample_counts.update(d.get("sample_counts", {}))
        return fv

    def pretty(self) -> str:
        lines = []
        for ax in AXES:
            v = self.scores.get(ax)
            n = self.sample_counts.get(ax, 0)
            bar = ("#" * int(round((v or 0) * 30))).ljust(30)
            lines.append("%-16s %s %.3f (n=%d)" % (ax, bar, v if v is not None else float("nan"), n))
        comp = self.composite()
        lines.append("-" * 56)
        lines.append("%-16s %s %.3f" % ("composite", "", comp if comp is not None else float("nan")))
        return "\n".join(lines)
