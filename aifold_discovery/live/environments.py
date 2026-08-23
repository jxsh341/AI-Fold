"""AI-Fold Live: Real environments with deterministic local verifiers.

Four environments expose the Atropos duck-type interface
(get_next_item / collect_trajectories) so the existing AtroposEnvAdapter
drives them unchanged â€” mock, atropos-native and live envs all share one
substrate path. Scores are computed locally (no judge model):

    math.reasoning     generated multi-step arithmetic/algebra word problems
    coding.execution   function synthesis scored by executing unit tests
    memory.long_context  needle-in-a-haystack; working_memory_size controls
                       how much context the scaffold assembles -> REAL
                       structural effect of memory genes on score
    agent.selfcorrection trap problems with plausible wrong answers;
                       measures detect-and-recover rate

All tasks embed a machine-readable JSON block so scaffolds can parse the
task type; truth never enters the prompt.
"""

import json
import re
import random
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from aifold_discovery.live.scaffolding import (
    GenomeScaffold, Task, ScaffoldResult, score_answer, _norm_number,
)

DIFFICULTY_SCALE = {"easy": 0.6, "medium": 1.0, "hard": 1.6}


# ======================================================================
# Shared base


class LiveBaseEnv:
    """Atropos-compatible surface for one live environment."""

    name = "live"
    capability_axes = ["reasoning"]
    difficulty = "medium"
    group_size = 4

    def __init__(self):
        self._iter = 0

    # -- interface expected by AtroposEnvAdapter -----------------------
    async def get_next_item(self) -> Dict[str, Any]:
        item = {"seed": 1000 + self._iter, "env": self.name}
        self._iter += 1
        return item

    async def collect_trajectories(self, item) -> Tuple[Optional[Dict], List]:
        scaffold: GenomeScaffold = item["scaffold"]
        n = self.group_size
        scores: List[float] = []
        tags = {"self_corrected": False, "tool_calls": 0}
        messages_out = []
        for i in range(n):
            seed = item["seed"] * 100 + i
            task = self.generate_task(seed)
            res = await scaffold.solve(task)
            score, correct = self.verify(task, res)
            scores.append(score)
            tags["tool_calls"] += res.tool_calls
            if correct is False and res.self_corrected and score > 0:
                tags["self_corrected"] = True
            scaffold.remember(task, res, correct)
            messages_out.append([
                {"role": "system", "content": "agent transcript"},
                {"role": "user", "content": task.prompt[:600]},
                {"role": "assistant", "content":
                    f"answer={res.answer} calls={res.n_llm_calls} "
                    f"tools={res.tool_calls} retried={res.retried}"},
            ])
        if not scores or len(set(scores)) == 1:
            return None, []
        return {
            "tokens": [[] for _ in scores],          # RL-tokenization offline
            "masks": [[0] for _ in scores],
            "scores": scores,
            "messages": messages_out,
            "group_overrides": {
                "group_size": len(scores),
                "self_corrected": tags["self_corrected"],
                "tool_calls": tags["tool_calls"],
            },
        }, []

    # -- subclass hooks -------------------------------------------------
    def generate_task(self, seed: int) -> Task:
        raise NotImplementedError

    def verify(self, task: Task, res: ScaffoldResult) -> Tuple[float, Optional[bool]]:
        s = score_answer(task, res.answer)
        return s, s > 0


# ======================================================================
# math.reasoning


class LiveMathEnv(LiveBaseEnv):
    name = "math.reasoning.v4"
    capability_axes = ["reasoning"]

    def generate_task(self, seed: int) -> Task:
        r = random.Random(seed)
        k = int(2 * DIFFICULTY_SCALE.get(self.difficulty, 1.0)) + 1   # steps
        nums = [r.randint(12, 99) for _ in range(k)]
        ops = [r.choice(["+", "-", "*"]) for _ in range(k - 1)]
        # build expression left-to-right
        expr = str(nums[0])
        truth = float(nums[0])
        story = [f"You start with {nums[0]} crates."]
        for op, n in zip(ops, nums[1:]):
            if op == "+":
                truth += n
                story.append(f"Then you receive {n} more.")
            elif op == "-":
                truth -= n
                story.append(f"Then you ship out {n}.")
            else:
                truth *= n
                story.append(f"Then production triples-scale by {n}x "
                             "(multiply current amount).")
            expr += f" {op} {n}"
        truth = int(truth)
        body = " ".join(story)
        prompt = (f"{body} How many crates are there in the end?\n\n"
                  "{\n"
                  f'  "type": "math",\n'
                  f'  "expr": "{expr}",\n'
                  f'  "difficulty": "{self.difficulty}"\n'
                  "}\n"
                  "Reason step by step, then reply FINAL_ANSWER: <number>")
        return Task(type="math", prompt=prompt, truth=truth,
                    difficulty=self.difficulty,
                    meta={"expr": expr})


# ======================================================================
# coding.execution


_CODE_PROBLEMS = [
    {
        "slug": "digit_sum",
        "docstring": "Return the sum of the decimal digits of non-negative integer n.",
        "tests": [("digit_sum(0)", 0), ("digit_sum(9)", 9),
                  ("digit_sum(1234)", 10), ("digit_sum(99999)", 45)],
    },
    {
        "slug": "second_max",
        "docstring": "Given a list of ints (len>=2), return the second largest DISTINCT value.",
        "tests": [("second_max([1, 2])", 1), ("second_max([5, 5, 3])", 3),
                  ("second_max([10, 30, 20])", 20), ("second_max([-1, -5, -3])", -3)],
    },
    {
        "slug": "count_vowels",
        "docstring": "Return the number of vowels (aeiou) in string s, case-insensitive.",
        "tests": [('count_vowels("")', 0), ('count_vowels("xyz")', 0),
                  ('count_vowels("Alpha Beta")', 5), ('count_vowels("AEIOUaeiou")', 10)],
    },
    {
        "slug": "flatten_once",
        "docstring": "Given a list of lists (one level deep), return the flattened list.",
        "tests": [('flatten_once([[1], [2, 3]])', "[1, 2, 3]"),
                  ('flatten_once([])', "[]"),
                  ('flatten_once([[], [7]])', "[7]")],
    },
]


class LiveCodeEnv(LiveBaseEnv):
    name = "coding.execution.v3"
    capability_axes = ["coding"]
    difficulty = "medium"

    def generate_task(self, seed: int) -> Task:
        r = random.Random(seed)
        prob = _CODE_PROBLEMS[r.randrange(len(_CODE_PROBLEMS))]
        test_lines = "\n".join(
            f"assert {t} == {json.dumps(exp)}" for t, exp in prob["tests"])
        harness = f"""
{prob['docstring']}
Function name and tests:
{test_lines}
Write ONLY a python code block defining `{prob['slug']}`.
"""
        meta = {"problem": prob["slug"],
                "tests": prob["tests"],
                "difficulty": self.difficulty}
        payload = json.dumps({"type": "code",
                              "task": prob["slug"],
                              "difficulty": self.difficulty})
        prompt = (harness + f"\n{{\n  \"meta\": {json.dumps(meta)}\n}}\n"
                  "Reply with one ```python block; no prose needed.")
        # truth hidden from model; used only via execution
        return Task(type="code", prompt=prompt, truth=prob["slug"],
                    difficulty=self.difficulty, meta=meta)

    def verify(self, task: Task, res: ScaffoldResult) -> Tuple[float, Optional[bool]]:
        from aifold_discovery.live.scaffolding import CODE_FENCE
        # Scan every transcript stage for a python block (solve + tool rounds).
        code = None
        for entry in reversed(res.transcript):
            blocks = CODE_FENCE.findall(entry.get("text", ""))
            if blocks:
                code = blocks[0]
                break
        if code is None:
            m = re.search(r"def\s+\w+.*?(?:return\s+.+|\n)", res.answer or "", re.S)
            code = res.answer if ("def " in (res.answer or "")) else None
        if not code or task.type != "code":
            return -1.0, False
        slug = task.truth
        checks = "\n".join(f"assert {t} == {json.dumps(e)}"
                           for t, e in task.meta["tests"])
        full = (f"{code}\n\n{checks}\nprint('ALL_TESTS_PASS')")
        try:
            proc = subprocess.run([sys.executable, "-c", full],
                                  capture_output=True, text=True, timeout=10)
            ok = proc.returncode == 0 and "ALL_TESTS_PASS" in proc.stdout
            return (1.0 if ok else -1.0), ok
        except subprocess.TimeoutExpired:
            return -1.0, False


# ======================================================================
# memory.long_context


class LiveMemoryEnv(LiveBaseEnv):
    name = "memory.long_context.v2"
    capability_axes = ["memory", "reasoning"]
    difficulty = "hard"

    def __init__(self):
        super().__init__()
        self.last_chunk_count = 0

    def generate_task(self, seed: int) -> Task:
        r = random.Random(seed)
        n_facts = int(24 * DIFFICULTY_SCALE.get(self.difficulty, 1.0))
        names = ["Zorith", "Kellam", "Vex", "Maro", "Thula", "Bren",
                 "Osk", "Lirien", "Dax", "Yewna"]
        items = ["obsidian compass", "brass key", "salt map", "iron lantern",
                 "glass feather", "bone whistle", "copper ring", "ash codex"]
        places = ["the drowned archive", "vault nine", "the pale terrace",
                  "harbor seventeen", "the glass steppe"]
        facts, truths = [], []
        for _ in range(n_facts):
            who = r.choice(names)
            what = r.choice(items)
            where = r.choice(places)
            facts.append((who, what, where))
            truths.append((who, what))
        target_who, target_what = truths[r.randrange(len(truths))]
        target_where = next(w for (a, b, w) in facts
                            if a == target_who and b == target_what)
        r.shuffle(facts)
        lines = [f"{a} last kept {b} at {w}." for a, b, w in facts]

        # working-memory gene controls how many lines fit into context.
        wm = max(2, self._current_working_mem)
        included = lines[:wm] if wm < len(lines) else lines
        self.last_chunk_count = min(wm, len(lines))

        prompt = ("FACTS:\n" + "\n".join(included) +
                  ("\n... (additional records not shown)\n" if wm < len(lines) else "") +
                  f"\nWhere did {target_who} last keep the {target_what}?\n\n" +
                  "{\n" +
                  f'  "type": "memory",\n'
                  f'  "needles_total": {len(lines)},\n'
                  f'  "context_lines_shown": {len(included)}\n' +
                  "}\nReply FINAL_ANSWER: <place name>")
        return Task(type="memory", prompt=prompt, truth=target_where,
                    difficulty=self.difficulty,
                    meta={"shown": len(included), "total": len(lines)})

    # set externally per-candidate by the runner/scaffold wrapper
    _current_working_mem: int = 8

    def set_candidate_memory(self, genome):
        self._current_working_mem = genome.memory.working_memory_size

    async def collect_trajectories(self, item):
        scaffold: GenomeScaffold = item["scaffold"]
        self.set_candidate_memory(scaffold.g)
        return await super().collect_trajectories(item)


# ======================================================================
# agent.selfcorrection


class LiveSelfCorrectionEnv(LiveBaseEnv):
    name = "agent.selfcorrection.v2"
    capability_axes = ["self_correction", "reasoning"]
    difficulty = "medium"

    def generate_task(self, seed: int) -> Task:
        """Order-of-operations / off-by-one style traps."""
        r = random.Random(seed)
        kind = r.choice(["order_of_ops", "inclusive_range", "rate_mixup"])
        if kind == "order_of_ops":
            a, b, c = r.randint(2, 9), r.randint(2, 9), r.randint(2, 9)
            truth = a + b * c
            trap = (a + b) * c
            text = (f"Compute: {a} + {b} Ã— {c}. Remember operator precedence.")
        elif kind == "inclusive_range":
            lo, hi = r.randint(1, 5), r.randint(8, 15)
            truth = hi - lo + 1
            trap = hi - lo
            text = (f"How many integers are in the inclusive range "
                    f"[{lo}, {hi}]?")
        else:
            dist, t_hours = r.choice([(60, 3), (90, 2), (120, 5), (45, 3)])
            truth = dist // t_hours
            trap = dist * t_hours
            text = (f"A courier travels {dist} km in {t_hours} hours at "
                    f"constant speed. What is the speed in km/h?")
        prompt = (text + "\n\n{\n" +
                  f'  "type": "selfcorrection",\n'
                  f'  "stage": "solve",\n'
                  f'  "trap_class": "{kind}"\n' +
                  "}\nThink carefully, then reply FINAL_ANSWER: <number>")
        t = Task(type="selfcorrection", prompt=prompt, truth=truth,
                 difficulty=self.difficulty, meta={"trap": str(trap)})
        return t

    def verify(self, task: Task, res: ScaffoldResult) -> Tuple[float, Optional[bool]]:
        ans = _norm_number(res.answer)
        correct = ans == _norm_number(task.truth)
        trapped = ans == _norm_number(task.meta.get("trap"))
        if correct and res.self_corrected:
            return 1.0, True          # recovered: strongest signal
        if correct:
            return 1.0, True
        if trapped:
            return -1.0, False        # fell for the planted trap
        return -1.0, False


ENV_CLASSES = {
    "math.reasoning.v4": LiveMathEnv,
    "coding.execution.v3": LiveCodeEnv,
    "memory.long_context.v2": LiveMemoryEnv,
    "agent.selfcorrection.v2": LiveSelfCorrectionEnv,
}


def make_env(registry_id: str):
    cls = ENV_CLASSES.get(registry_id)
    if cls is None:
        raise KeyError(f"no live env for {registry_id}")
    return cls()

