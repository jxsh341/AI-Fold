"""AI-Fold LIVE: evolutionary discovery against a real LLM.

    python run_live_discovery.py [--generations 2] [--pop-size 6]
                                 [--group-size 4] [--max-calls 1500]
                                 [--model ...] [--experiments-only]

Reads .env (AIFOLD_BASE_URL / AIFOLD_API_KEY / AIFOLD_MODEL).
Default backend: NVIDIA NIM (integrate.api.nvidia.com).

What runs live:
  - real LLM calls via OpenAI-compatible backend
  - genome scaffolding executes for real: decomposition, verification
    passes, critic gates, sandboxed code tool, episodic memory injection
  - four verifiable environments scored locally (no judge model)
  - evidence -> multi-dimensional fitness -> targeted mutation/crossover
  - Pareto + novelty selection, scientific archive persisted
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "aifold_discovery"))


def load_dotenv_simple(path: str = ".env"):
    p = ROOT / path
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def event_logger(kind: str, payload: dict):
    if kind == "population_seeded":
        print(f"[seed] population of {payload['size']} candidates")
    elif kind == "child_created":
        tgt = payload.get("targeted_axis") or "-"
        print(f"  [child {payload['child']}] <- {payload['parents']} "
              f"via '{payload['mutation']}' (targeted: {tgt})")
    elif kind == "experiment_done":
        vp = payload.get("vs_parent")
        base = f"    [exp] {payload['genome']} @ {payload['env']}"
        if vp is not None:
            print(f"{base}: vs-parent {vp:+.3f}")
        else:
            deltas = {k: v for k, v in payload["delta"].items() if abs(v) > 1e-9}
            dstr = ", ".join(f"{k}{v:+.2f}" for k, v in list(deltas.items())[:4]) or "measured"
            print(f"{base}: {dstr}")
        if payload["diagnosis"] and "no significant" not in payload["diagnosis"]:
            print(f"          {payload['diagnosis']}")
    elif kind == "generation_best":
        f = payload.get("fitness")
        fs = f"{f:.3f}" if isinstance(f, float) else "n/a"
        print(f"[gen best] {payload['genome']} fitness={fs} :: {payload['desc']}")
    elif kind == "generation_end":
        print(f"[gen end] {payload['duration_s']}s, "
              f"{payload['new_children']} children created")
    else:
        print(f"[{kind}] {str(payload)[:160]}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--pop-size", type=int, default=6)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--n-envs", type=int, default=2,
                    help="environments evaluated per candidate per generation")
    ap.add_argument("--max-calls", type=int, default=1200,
                    help="global LLM call budget")
    ap.add_argument("--max-concurrency", type=int, default=4)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--difficulty", type=str, default=None,
                    choices=["easy", "medium", "hard"],
                    help="override all env difficulties (headroom control)")
    ap.add_argument("--pure-baseline", action="store_true",
                    help="seed ONLY plain baselines: force evolution to "
                         "discover structure (verifier etc.) itself")
    args = ap.parse_args()

    load_dotenv_simple()
    from aifold_discovery.live.backend import detect_backend, OpenAICompatBackend
    backend = await detect_backend(
        override_model=args.model, allow_mock=False,
    )
    if isinstance(backend, OpenAICompatBackend):
        backend._sem = asyncio.Semaphore(args.max_concurrency)
        backend.max_retries = 5
        print(f"[backend] {backend.name} "
              f"(concurrency={args.max_concurrency}, retries<=5 w/ backoff)")
    else:
        print(f"[backend] {backend.name}")

    from aifold_discovery.atropos_bridge.adapter import AtroposEnvAdapter
    # Patch run_environment path: registry specs already carry env_factory;
    # adapter.build_adapter will construct live envs through make_env.
    from aifold_discovery.search.evolutionary import (
        DiscoveryEngine, RLTrainHook,
    )
    from aifold_discovery.live.registry_live import build_live_registry
    from aifold_discovery.live.trainer import LiveRLHook

    budget = {"left": args.max_calls}

    class BudgetedBackend:
        """Wraps real backend; hard-stops at the call budget."""

        def __init__(self, inner):
            self.inner = inner
            self.name = inner.name

        async def chat(self, messages, **kw):
            if budget["left"] <= 0:
                raise RuntimeError("LLM call budget exhausted")
            budget["left"] -= 1
            return await self.inner.chat(messages, **kw)

        def usage_summary(self):
            return self.inner.usage_summary()

    bbackend = BudgetedBackend(backend)

    # Route env_factory construction through the shared backend by making
    # make_env-produced envs receive scaffold-per-candidate at collect time.
    # AtroposEnvAdapter.run(genome) builds GenomeScaffold(genome, bbackend)
    # and passes it inside items — see monkeypatch below.
    import aifold_discovery.atropos_bridge.adapter as adapter_mod
    from aifold_discovery.live.scaffolding import GenomeScaffold

    orig_run = AtroposEnvAdapter.run

    async def live_run(self, genome, n_groups=2, group_size=None, **kw):
        env = self.env
        gs = group_size or getattr(getattr(env, "config", None),
                                   "group_size", self.spec.group_size)
        scaffold = GenomeScaffold(genome, bbackend)
        evidences = []
        for _g in range(n_groups):
            item = await env.get_next_item()
            item["scaffold"] = scaffold
            try:
                group, _bl = await env.collect_trajectories(item)
            except RuntimeError:
                raise                      # budget exhausted -> abort run
            except Exception:
                continue                   # env failure -> no evidence
            if not group or not isinstance(group, dict):
                continue
            # diagnostics flow inside group_overrides (see LiveBaseEnv)
            evidences.append(adapter_mod.TrajectoryEvidence.from_atropos_group(
                group, self.spec.registry_id))
        return evidences

    AtroposEnvAdapter.run = live_run

    registry = build_live_registry(group_size=args.group_size,
                                   difficulty_override=args.difficulty)
    engine = DiscoveryEngine(
        registry=registry,
        pop_size=args.pop_size,
        seed=11,
        rl_hook=LiveRLHook(enabled=True),
        groups_per_experiment=1,       # 1 group x group_size episodes per exp
        on_event=event_logger,
    )

    t0 = time.time()
    engine.seed_population(n_variants=1 if args.pure_baseline else 3)
    gen_history = []

    sem = asyncio.Semaphore(2)          # two candidates in flight
    budget_dead = {"flag": False}

    async def eval_one(cand):
        if cand.status == "evaluated" or budget_dead["flag"]:
            return
        async with sem:
            if budget_dead["flag"]:
                return
            try:
                await engine.evaluate_candidate(cand, n_envs=args.n_envs)
            except RuntimeError as e:               # LLM budget exhausted
                if "budget" in str(e).lower():
                    budget_dead["flag"] = True
                    print(f"[budget] exhausted — stopping new evaluations ({e})")
                else:
                    raise

    for g in range(args.generations):
        print(f"\n{'=' * 72}\nLIVE GENERATION {g}\n{'=' * 72}")
        await asyncio.gather(*[eval_one(c) for c in list(engine.population.all())])
        if budget_dead["flag"]:
            break
        await engine.maybe_rl_train(engine.population.best())
        best = engine.population.best()
        f = best.composite_fitness() if best else None
        gen_history.append((g, best.genome_id if best else None, f))
        print(f"[gen best] {best.describe() if best else '-'} fitness="
              f"{f:.3f}" if f is not None else "[gen best] none")
        from aifold_discovery.memory.archive import DiscoveryArchive
        DiscoveryArchive.persist_run(engine, tag=f"live_gen{g}")
        children = engine.next_generation_candidates(
            n_children=max(2, len(engine.population.all()) // 2))
        for ch in children:
            engine.population.add(ch)

    # ---------------- final report ----------------
    best = engine.population.best()
    print(f"\n{'=' * 72}\nBEST LIVE CANDIDATE\n{'=' * 72}")
    print(best.describe())
    print(best.fitness.pretty())

    stats = engine.store.discovery_stats()
    print("\nMutation effectiveness (child composite - strongest parent, avg):")
    for k, v in sorted(stats["avg_delta_by_mutation"].items(),
                       key=lambda t: -t[1]):
        print(f"  {k:34s} {v:+.4f}")
    print("\nEnvironment difficulty (avg pass rate):")
    for k, v in sorted(stats["avg_pass_rate_by_env"].items(),
                       key=lambda t: -t[1]):
        print(f"  {k:34s} {v:.3f}")

    usage = backend.usage_summary()
    print(f"\n[usage] calls={usage.get('calls', 0)} "
          f"in={usage.get('prompt_tokens', 0)}tok "
          f"out={usage.get('completion_tokens', 0)}tok "
          f"errors={usage.get('errors', 0)} "
          f"budget_left={budget['left']} "
          f"wall={time.time() - t0:.0f}s")

    if gen_history:
        print("\n[cross-generation best-composite trajectory]")
        for g, gid, f in gen_history:
            fs = f"{f:.3f}" if isinstance(f, float) else "n/a"
            print(f"  generation {g}: best={gid} composite={fs}")

    from aifold_discovery.memory.archive import DiscoveryArchive
    arch, paths = DiscoveryArchive.persist_run(engine, tag="live")
    print("\nArchived:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    asyncio.run(main())
