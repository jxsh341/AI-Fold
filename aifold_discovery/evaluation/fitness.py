"""AI-Fold Discovery: Fitness Attribution.

Converts TrajectoryEvidence collected from Atropos environments into
FitnessVector updates. This is where scalar RL rewards become a
multi-dimensional capability profile.

Attribution rules:
  - Each environment declares its capability axes; evidence pass_rate
    is the primary per-axis measurement.
  - score_variance acts as a learning-signal quality weight: groups
    where everything passes or everything fails carry less information.
  - self-correction recovery and tool-call intensity are attributed to
    their dedicated axes from trajectory tags.
  - efficiency derives from token cost relative to success.
  - generalization/robustness derive from held-out vs train deltas when
    both are available.
"""

from typing import Dict, List, Optional, Tuple

from aifold_discovery.core.fitness import FitnessVector, AXES
from aifold_discovery.core.experiment import ExperimentRecord, TrajectoryEvidence


def _evidence_axis_weight(ev: TrajectoryEvidence) -> float:
    """Information content of this group: variance-rich groups count more."""
    # Map variance in [0, ~1] to [0.5, 1.0]; all-same-scored groups get 0.5.
    return 0.5 + min(1.0, ev.score_variance)


def _pass_rate_score(ev: TrajectoryEvidence) -> float:
    """Map raw scores to [0,1]. Atropos uses +1 / -1 conventions."""
    if ev.n_items == 0:
        return 0.0
    pos = sum(1 for s in ev.scores if s > 0)
    return pos / ev.n_items


def attribute_record(record: ExperimentRecord,
                     baseline: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Extract per-axis measurements from one experiment record."""
    baseline = baseline or {}
    axis_scores: Dict[str, Tuple[float, float]] = {}   # axis -> (weighted_sum, weight_sum)

    def add(axis: str, value: float, weight: float):
        s, w = axis_scores.get(axis, (0.0, 0.0))
        axis_scores[axis] = (s + value * weight, w + weight)

    for ev in record.evidences:
        if ev.n_items == 0:
            continue
        pr = _pass_rate_score(ev)
        iw = _evidence_axis_weight(ev)

        # Primary axes declared by the environment spec
        for axis in record_axes(record):
            add(axis, pr, iw * 1.0)

        # Dedicated signals from tags
        if ev.had_self_correction:
            # recovery: fraction of initially-failed items later passing is
            # approximated by pass_rate uplift in selfcorrection envs
            add("self_correction", max(pr, 0.0), iw * 1.2)
        if ev.tool_call_count > 0:
            tool_eff = pr * min(1.0, ev.tool_call_count / max(1, ev.tool_call_count))
            add("tool_use", pr, iw)
            add("tool_use", tool_eff, iw * 0.3)

        # Efficiency: successful groups that are short get rewarded
        if ev.mean_length_tokens > 0:
            length_norm = 1.0 / (1.0 + ev.mean_length_tokens / 2048.0)
            add("efficiency", pr * (0.6 + 0.4 * length_norm), iw * 0.8)

        # Robustness: performance when failures were present in-group
        if ev.had_failures:
            add("robustness", pr, iw * 0.9)

    out = {}
    for ax, (s, w) in axis_scores.items():
        if w > 0:
            out[ax] = s / w
    return out


def record_axes(record: ExperimentRecord) -> List[str]:
    """Capability axes attached to this record's environment entry."""
    ax = getattr(record, "env_capability_axis", "") or ""
    if "," in ax:
        return [a.strip() for a in ax.split(",") if a.strip()]
    return [ax] if ax else []


def update_fitness(fitness: FitnessVector,
                   record: ExperimentRecord,
                   alpha: float = 0.35) -> FitnessVector:
    """Running-mean fitness update from one experiment record."""
    measured = attribute_record(record)
    updated = FitnessVector.from_dict(fitness.to_dict())
    for ax, val in measured.items():
        if ax not in AXES:
            continue
        cur = updated.get(ax)
        if cur is None:
            updated.set(ax, val, n_samples=sum(e.n_items for e in record.evidences))
        else:
            merged_val = alpha * val + (1 - alpha) * cur
            updated.set(ax, merged_val,
                        n_samples=max(1, sum(e.n_items for e in record.evidences)))
    return updated


# ----------------------------------------------------------------------
# Diagnosis: turn fitness deltas into human-readable failure analysis.


DIAGNOSIS_RULES = [
    ("memory", "insufficient long-context retention; consider episodic memory"),
    ("planning", "poor multi-step decomposition; consider planner/search"),
    ("tool_use", "under-utilizing tools; widen tool access or budget"),
    ("self_correction", "no effective verify-and-retry loop; enable verifier/critic"),
    ("coding", "weak code synthesis/execution; consider code tool"),
    ("reasoning", "shallow reasoning chains; consider verifier or deeper search"),
    ("efficiency", "over-computation relative to success; simplify control flow"),
    ("robustness", "brittle under partial failure; enable retries/critic"),
    ("generalization", "train-eval gap; diversify training environments"),
]


def diagnose(record: ExperimentRecord,
             pre_fitness: FitnessVector,
             post_fitness: FitnessVector) -> str:
    """Produce a short diagnosis string from before/after fitness."""
    regressions = []
    for ax in AXES:
        pre, post = pre_fitness.get(ax), post_fitness.get(ax)
        if pre is None or post is None:
            continue
        if post < pre - 0.02:
            regressions.append((ax, pre - post))
    regressions.sort(key=lambda t: -t[1])
    if not regressions:
        improved = [
            (ax, post_fitness.get(ax) - pre_fitness.get(ax))
            for ax in AXES
            if pre_fitness.get(ax) is not None and post_fitness.get(ax) is not None
            and post_fitness.get(ax) > pre_fitness.get(ax) + 0.02
        ]
        if improved:
            improved.sort(key=lambda t: -t[1])
            ax, d = improved[0]
            return f"improved {ax} (+{d:.3f}); mutation validated"
        return "no significant change"

    worst_ax, worst_d = regressions[0]
    for ax, _d in regressions:
        for rule_ax, text in DIAGNOSIS_RULES:
            if rule_ax == ax:
                return text
    return f"regression on {worst_ax}"
