"""AI-Fold Discovery: end-to-end demo.

Runs the complete absorbed-Atropos loop in dry-run mode:

    seed population -> evolve N generations -> archive scientific history

Environments run through the MockEnvAdapter (capability-prior model), so
this requires no LLM server and no Atropos deployment. It validates the
full mechanics:

    - genome mutation/crossover with fitness-targeted hypotheses
    - evidence-balanced environment scheduling
    - trajectory evidence -> multi-dimensional fitness attribution
    - Pareto + novelty selection
    - experiment records with lineage and diagnosis
    - RL boundary respected (rl_hook disabled => no weight updates)

To go live: pass rl_hook=RLTrainHook(enabled=True) wired to the Atropos
trainer, and register EnvironmentSpecs whose env_factory returns real
atroposlib BaseEnv instances.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aifold_discovery.search.evolutionary import DiscoveryEngine
from aifold_discovery.atropos_bridge.registry import EnvironmentRegistry
from aifold_discovery.memory.archive import DiscoveryArchive


def event_logger(kind: str, payload: dict):
    """Compact console trace of the discovery loop."""
    if kind == "population_seeded":
        print(f"[seed] population of {payload['size']} candidates")
    elif kind == "child_created":
        tgt = payload.get("targeted_axis") or "-"
        print(f"  [child {payload['child']}] <- {payload['parents']} "
              f"via '{payload['mutation']}' (targeted axis: {tgt})")
    elif kind == "experiment_done":
        deltas = {k: v for k, v in payload["delta"].items() if v != 0}
        dstr = ", ".join(f"{k}{v:+.2f}" for k, v in list(deltas.items())[:4]) or "no change"
        print(f"    [exp] {payload['genome']} @ {payload['env']}: {dstr}")
        if payload["diagnosis"]:
            print(f"          diagnosis: {payload['diagnosis']}")
    elif kind == "generation_best":
        f = payload.get("fitness")
        fs = f"{f:.3f}" if isinstance(f, float) else "n/a"
        print(f"[gen best] {payload['genome']} fitness={fs} :: {payload['desc']}")
    elif kind == "generation_end":
        print(f"[gen end] {payload['duration_s']}s, {payload['new_children']} new children")
    elif kind == "discovery_complete":
        print("\n=== DISCOVERY COMPLETE ===")
    else:
        print(f"[{kind}] {payload}")


async def main(generations: int = 3, pop_size: int = 10):
    engine = DiscoveryEngine(
        registry=EnvironmentRegistry.default(),
        pop_size=pop_size,
        seed=7,
        groups_per_experiment=2,
        on_event=event_logger,
    )

    engine.seed_population(n_variants=3)

    for g in range(generations):
        print(f"\n{'=' * 72}")
        print(f"GENERATION {g}")
        print(f"{'=' * 72}")
        await engine.run_generation(n_envs_per_candidate=2)

    # Final report -----------------------------------------------------
    best = engine.population.best()
    print(f"\n{'=' * 72}")
    print("BEST CANDIDATE")
    print(f"{'=' * 72}")
    print(best.describe())
    print(best.fitness.pretty())

    front = engine.population.pareto_front()
    print(f"\nPareto front ({len(front)} specialists preserved):")
    for c in front:
        comp = c.composite_fitness()
        cs = f"{comp:.3f}" if comp is not None else "n/a"
        print(f"  {c.describe()}  [composite={cs}]")

    stats = engine.store.discovery_stats()
    print("\nWhat the search learned:")
    print("  avg fitness delta by mutation:")
    for k, v in sorted(stats["avg_delta_by_mutation"].items(), key=lambda t: -t[1]):
        print(f"    {k:32s} {v:+.4f}")
    print("  avg pass rate by environment:")
    for k, v in sorted(stats["avg_pass_rate_by_env"].items(), key=lambda t: -t[1]):
        print(f"    {k:32s} {v:.3f}")

    # Persist the scientific history -----------------------------------
    arch, paths = DiscoveryArchive.persist_run(engine, root="./aifold_runs", tag="demo")
    arch.save_summary({
        "best": best.to_dict(),
        "pareto_front": [c.describe() for c in front],
        "stats": stats,
    })
    print("\nArchived:")
    for k, p in paths.items():
        print(f"  {k}: {p}")

    return engine


if __name__ == "__main__":
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    asyncio.run(main(generations=gens))
