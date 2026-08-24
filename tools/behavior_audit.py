"""Behavioral phenotype audit: consumed genes must produce their DOCUMENTED
observable difference, not merely appear in a call graph.

Each check builds its own OFF/ON genome pair, runs GenomeScaffold against a
deterministic scripted backend (no network), and asserts the documented
diagnostic difference. Exit code 1 on any FAIL.
"""
import sys, asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'aifold_discovery'))

from aifold_discovery.core.genome import baseline_genome
from aifold_discovery.evolution.mutation import MUTATION_INDEX, Mutator
from aifold_discovery.live.scaffolding import GenomeScaffold, Task
from aifold_discovery.live.backend import LLMBackend


class R:
    def __init__(self, text):
        self.text = text
        self.error = None
        self.prompt_tokens = 0
        self.completion_tokens = 0


class ScriptedBackend(LLMBackend):
    name = "scripted"

    def __init__(self, fail_first_n: int = 0):
        self.calls = 0
        self.user_contents = []
        self.fail_first_n = fail_first_n

    async def chat(self, messages, temperature=0.7, max_tokens=1024, stop=None):
        self.calls += 1
        user = next((m["content"] for m in reversed(messages)
                     if m.get("role") == "user"), "")
        sysc = next((m["content"] for m in messages
                     if m.get("role") == "system"), "")
        self.user_contents.append(user)

        if self.calls <= self.fail_first_n:
            return R("")
        if "strict verifier" in sysc:
            return R("VERDICT: CORRECT\nFINAL_ANSWER: 42")
        if "You are a critic" in sysc:
            return R("CRITIC: PASS")
        if "planning module" in sysc:
            return R("1. read\n2. compute")
        if "Refine your answer" in sysc:
            return R("FINAL_ANSWER: 42")
        if "```python" in user:
            return R("```python\nx = 6*7\nprint(x)\n```\nFINAL_ANSWER: 42")
        return R("FINAL_ANSWER: 42")


def make_task(difficulty="medium", long=False):
    pad = "x" * (1500 if long else 10)
    return Task(type="math",
                prompt=f"Solve. {{\"type\": \"math\"}}\n{pad}",
                truth=42, difficulty=difficulty)


def apply(gene, g):
    MUTATION_INDEX[gene][2](g)
    return g


async def run_pair(g_off, g_on, task=None, fail_first_n=0):
    outs = []
    for g in (g_off, g_on):
        be = ScriptedBackend(fail_first_n=fail_first_n)
        sc = GenomeScaffold(g, be, max_llm_calls=40)
        r = await sc.solve(task or make_task())
        outs.append((r, be))
    return outs


# ---------------- checks ----------------

CHECKS = {}


def check(name, doc):
    def deco(fn):
        CHECKS[name] = (doc, fn)
        return fn
    return deco


@check("add_verifier", "verify stage fires -> extra LLM call + verified flag")
async def _():
    o, n = await run_pair(baseline_genome(),
                          apply("add_verifier", baseline_genome()))
    return n[0].verified and not o[0].verified and n[0].n_llm_calls > o[0].n_llm_calls


@check("enable_critic", "critic gate runs as an additional stage")
async def _():
    o, n = await run_pair(baseline_genome(),
                          apply("enable_critic", baseline_genome()))
    return n[0].n_llm_calls > o[0].n_llm_calls


@check("conditional_router", "hard/long tasks route to deliberate path w/o verifier")
async def _():
    t = make_task(difficulty="hard", long=True)
    o, n = await run_pair(baseline_genome(),
                          apply("conditional_router", baseline_genome()),
                          task=t)
    routed_on = n[0].decomposed or n[0].verified
    routed_off = o[0].decomposed or o[0].verified
    return routed_on and not routed_off


@check("enable_decomposition", "PLAN step emitted before solving")
async def _():
    o, n = await run_pair(baseline_genome(),
                          apply("enable_decomposition", baseline_genome()))
    return n[0].decomposed and not o[0].decomposed


@check("upgrade_search", "none->bfs adds refinement round even at depth 1")
async def _():
    o, n = await run_pair(baseline_genome(),
                          apply("upgrade_search", baseline_genome()))
    return n[0].refinements > o[0].refinements and \
        n[0].n_llm_calls > o[0].n_llm_calls


@check("deepen_search", "search_depth scales refinement rounds")
async def _():
    off_g = apply("upgrade_search", baseline_genome())       # bfs @ depth 1
    on_g = apply("deepen_search", off_g.clone())             # bfs @ depth 3
    o, n = await run_pair(off_g, on_g)
    return n[0].refinements > o[0].refinements and \
        n[0].n_llm_calls > o[0].n_llm_calls


@check("add_tool_code", "code fences executed in sandbox only when tool present")
async def _():
    t = Task(type="code",
             prompt="Write ```python``` that computes 6*7.\n"
                    "{\"type\": \"code\", \"task\": \"digit_sum\"}",
             truth="digit_sum", difficulty="medium")
    o, n = await run_pair(baseline_genome(),
                          apply("add_tool_code", baseline_genome()), task=t)
    return n[0].tool_calls > o[0].tool_calls


@check("raise_tool_budget", "budget doubled when code tool present; refused otherwise")
async def _():
    g_codeless = baseline_genome()
    _, _, fn = MUTATION_INDEX["raise_tool_budget"]
    refused = fn(g_codeless) is None                     # no code -> refuses
    g_code = baseline_genome(); g_code.tools.enabled_tools.append("code")
    b0 = g_code.tools.max_tool_calls_per_episode
    applied = fn(g_code) is not None and \
        g_code.tools.max_tool_calls_per_episode > b0
    return refused and applied


@check("expand_working_memory", "memory env shows doubled context lines")
async def _():
    from aifold_discovery.live.environments import LiveMemoryEnv
    env = LiveMemoryEnv()
    g_off = baseline_genome()
    g_on = apply("expand_working_memory", baseline_genome())
    env.set_candidate_memory(g_off)
    s8 = env.generate_task(seed=777).meta["shown"]
    env.set_candidate_memory(g_on)
    s16 = env.generate_task(seed=777).meta["shown"]
    return s16 == 2 * s8 and s8 > 0


@check("add_episodic_memory", "cross-episode lessons injected into later prompts")
async def _():
    g_on = apply("add_episodic_memory", baseline_genome())
    results = {}
    for tag, cand in (("off", baseline_genome()), ("on", g_on)):
        be = ScriptedBackend()
        sc = GenomeScaffold(cand, be, max_llm_calls=40)
        t1 = make_task()
        r1 = await sc.solve(t1)
        sc.remember(t1, r1, correct=False)
        await sc.solve(make_task())
        results[tag] = any("RELEVANT MEMORY" in u for u in be.user_contents)
    return results["on"] and not results["off"]


@check("add_semantic_memory", "semantic memory enables injection path too")
async def _():
    g_on = apply("add_semantic_memory", baseline_genome())
    results = {}
    for tag, cand in (("off", baseline_genome()), ("on", g_on)):
        be = ScriptedBackend()
        sc = GenomeScaffold(cand, be, max_llm_calls=40)
        t1 = make_task()
        r1 = await sc.solve(t1)
        sc.remember(t1, r1, correct=False)
        await sc.solve(make_task())
        results[tag] = any("RELEVANT MEMORY" in u for u in be.user_contents)
    return results["on"] and not results["off"]


@check("enable_retries", "third failure recovered only with extra retry budget")
async def _():
    # Baseline: retry_on_failure=True, max_retries=2 -> survives 2 empty
    # responses. ON: retries=3 -> survives 3. Backend fails first 3 calls.
    o, n = await run_pair(baseline_genome(),
                          apply("enable_retries", baseline_genome()),
                          fail_first_n=3)
    recovered_on = n[0].retried and n[0].answer != ""
    exhausted_off = (not o[0].answer) or (o[0].retried and o[0].answer == "" and
                                          o[0].n_llm_calls < n[0].n_llm_calls)
    return recovered_on and (not o[0].answer or o[0].n_llm_calls < n[0].n_llm_calls) \
        and exhausted_off is not False


async def main():
    m = Mutator()
    base = baseline_genome()
    sampleable = m.applicable(base, allow_rl=False)

    missing = [g for g in sampleable if g not in CHECKS]
    extra = [k for k in CHECKS if k not in sampleable]
    if missing:
        print("MISSING BEHAVIORAL CHECKS FOR:", missing)
        sys.exit(2)

    failures = []
    for name, (doc, fn) in CHECKS.items():
        try:
            ok = await fn()
            print(f"[{'PASS' if ok else 'FAIL'}] {name:24s} {doc}")
            if not ok:
                failures.append(name)
        except Exception as e:
            print(f"[ERROR] {name}: {type(e).__name__}: {e}")
            failures.append(name)

    print(f"\ncoverage: checks cover all {len(sampleable)} live-sampleable genes"
          + (f"; extra checks: {extra}" if extra else ""))
    print(f"{len(CHECKS) - len(failures)}/{len(CHECKS)} behavioral checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    asyncio.run(main())