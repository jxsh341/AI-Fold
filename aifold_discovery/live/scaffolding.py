"""AI-Fold Live: Genome Scaffolding.

Turns a CandidateGenome into actual agent behavior against a live LLM
backend. This is where evolution becomes executable:

    planning.decomposition  -> multi-step plan-then-solve prompting
    planning.search (beam)  -> N parallel candidate solutions + vote
    model.verifier_enabled  -> explicit verify pass; retry with feedback
    control.critic_enabled  -> checklist critique gate before answering
    tools.code              -> model-written Python executed in a sandbox,
                               output fed back (budget-capped)
    memory.*                -> persistent cross-episode scratchpad injected
                               as retrieved context
    control.retry_on_failure-> bounded retries on parse/execution failure

Every scaffold returns rich diagnostics so evidence attributes onto the
correct fitness axes.
"""

import asyncio
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from aifold_discovery.core.genome import CandidateGenome
from aifold_discovery.live.backend import LLMBackend

CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)
FINAL = re.compile(r"FINAL_ANSWER\s*:\s*(.+?)\s*$", re.I | re.M)


@dataclass
class Task:
    """A verifiable unit of work handed to a scaffold."""
    type: str                 # math | code | memory | selfcorrection
    prompt: str               # full user-facing task text (includes JSON block)
    truth: Any = None         # ground-truth answer (local verifier uses this)
    difficulty: str = "medium"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScaffoldResult:
    answer: str
    n_llm_calls: int = 0
    tool_calls: int = 0
    retried: bool = False
    self_corrected: bool = False   # initial wrong -> final right (env-tagged)
    decomposed: bool = False
    verified: bool = False
    transcript: List[Dict[str, str]] = field(default_factory=list)
    error: Optional[str] = None


def extract_answer(text: str) -> str:
    """Robustly pull the final answer out of a model reply."""
    if not text:
        return ""
    vals = FINAL.findall(text or "")
    for raw in reversed(vals):
        v = raw.strip().strip("'\"").strip()
        if not v:
            continue
        # Model sometimes echoes the marker back inside the value.
        if v.upper().startswith("FINAL_ANSWER"):
            v = v.split(":", 1)[-1].strip()
        v = v.strip("*` ").rstrip(".").strip()
        if v:
            return v
    # Fallback: last number-ish token anywhere in the reply.
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""


def _norm_number(s: str):
    try:
        f = float(str(s).strip().replace(",", "").rstrip("."))
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return str(s).strip().lower()


class CodeSandbox:
    """Execute model-written python in an isolated subprocess."""

    def run(self, code: str, timeout_s: float = 10.0) -> Tuple[str, str]:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=timeout_s,
            )
            return proc.stdout.strip(), proc.stderr.strip()
        except subprocess.TimeoutExpired:
            return "", "TIMEOUT"


class GenomeScaffold:
    """Executable embodiment of one candidate genome."""

    SYSTEM_BASE = (
        "You are a precise problem-solving agent. "
        "Think step by step, then end your reply with exactly one line: "
        "FINAL_ANSWER: <answer>"
    )

    def __init__(self, genome: CandidateGenome, backend: LLMBackend,
                 max_llm_calls: int = 24):
        self.g = genome
        self.be = backend
        self.max_calls = max_llm_calls
        self.calls_used = 0
        self.sandbox = CodeSandbox()
        # Episodic memory: cross-episode lessons for THIS candidate.
        self.episodes: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    async def _chat(self, messages, temperature=0.4, max_tokens=900) -> Tuple[str, bool]:
        if self.calls_used >= self.max_calls:
            return "", False
        self.calls_used += 1
        r = await self.be.chat(messages, temperature=temperature,
                               max_tokens=max_tokens)
        return (r.text or ""), r.error is None

    # ------------------------------------------------------------------
    def _memory_block(self, task_type: str, k: int = 3) -> str:
        if not (self.g.memory.episodic_memory or self.g.memory.semantic_memory):
            return ""
        rel = [e for e in self.episodes if e["type"] == task_type][-k:]
        if not rel:
            return ""
        lines = ["RELEVANT MEMORY (lessons from earlier episodes):"]
        for e in rel[-self.g.memory.retrieval_k:]:
            tag = "succeeded" if e["correct"] else "failed"
            lines.append(f"- similar {e['type']} task {tag}. {e['note']}")
        return "\n".join(lines) + "\n"

    def remember(self, task: Task, result: ScaffoldResult, correct: bool):
        note = ("answer accepted" if correct
                else f"answered '{result.answer[:40]}' but truth was "
                     f"'{str(task.truth)[:40]}'; be careful with setup/computation")
        self.episodes.append({"type": task.type, "correct": correct, "note": note})
        cap = max(8, self.g.memory.working_memory_size * 2)
        if len(self.episodes) > cap:
            self.episodes = self.episodes[-cap:]

    # ------------------------------------------------------------------
    async def _plan(self, task: Task) -> List[str]:
        msgs = [
            {"role": "system", "content":
                "You are a planning module. Decompose the task into short "
                "ordered steps. Reply ONLY with numbered steps."},
            {"role": "user", "content": task.prompt},
        ]
        text, ok = await self._chat(msgs, temperature=0.2, max_tokens=400)
        steps = [ln.lstrip("0123456789.-) ").strip()
                 for ln in text.splitlines() if ln.strip()][:6]
        return steps if ok and steps else []

    # ------------------------------------------------------------------
    async def _solve_core(self, task: Task, context: str,
                          force_plan: bool = False) -> Tuple[str, ScaffoldResult]:
        """One solve attempt (with optional code tool). Returns (text, diag)."""
        res = ScaffoldResult(answer="")
        sys_prompt = self.SYSTEM_BASE
        user_parts = []
        if context:
            user_parts.append(context)

        if self.g.planning.decomposition or force_plan:
            steps = await self._plan(task)
            res.decomposed = bool(steps)
            res.n_llm_calls += 1
            if steps:
                user_parts.append("PLAN:\n" + "\n".join(
                    f"{i+1}. {s}" for i, s in enumerate(steps)))

        user_parts.append("TASK:\n" + task.prompt)
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": "\n\n".join(user_parts)}]

        text, ok = await self._chat(messages)
        res.n_llm_calls += 1
        res.transcript.append({"stage": "solve", "text": text[:800]})
        if not ok:
            res.error = "backend_error"
            return text, res

        # ---- code tool loop ------------------------------------------
        budget = (self.g.tools.max_tool_calls_per_episode
                  if self.g.tools.enabled_tools and "code" in self.g.tools.enabled_tools
                  else 0)
        tool_rounds = 0
        while tool_rounds < budget:
            blocks = CODE_FENCE.findall(text)
            if not blocks:
                break
            code = blocks[0]
            out, err = self.sandbox.run(code)
            res.tool_calls += 1
            tool_rounds += 1
            feedback = (f"CODE EXECUTION RESULT:\nstdout:\n{out}\nstderr:\n{err}"
                        if err else f"CODE EXECUTION RESULT:\n{out}")
            messages.append({"role": "assistant",
                             "content": text[-1500:]})
            messages.append({"role": "user", "content":
                             feedback + "\n\nUse this result. End with "
                             "FINAL_ANSWER: <answer>"})
            text, ok = await self._chat(messages, temperature=0.2)
            res.n_llm_calls += 1
            res.transcript.append({"stage": "tool_feedback", "text": text[:500]})
            if not ok:
                break

        res.answer = extract_answer(text)
        return text, res

    # ------------------------------------------------------------------
    async def solve(self, task: Task) -> ScaffoldResult:
        context = self._memory_block(task.type)

        # ---- conditional router: REAL routing decision ----------------
        # Deliberate path costs extra calls (plan + verify) but is granted
        # to hard/long tasks even without the verifier gene. This gives
        # control.router_type a genuine behavioral effect and a measurable
        # efficiency/correctness tradeoff.
        route_deliberate = False
        if self.g.control.router_type == "conditional":
            route_deliberate = (
                task.difficulty == "hard" or len(task.prompt) > 1200
            )
        deliberate = self.g.model.verifier_enabled or route_deliberate

        # Beam search: parallel candidates + majority vote on normalized answer
        beam = 1
        if self.g.planning.search_algorithm in ("beam", "mcts"):
            beam = 3

        attempts: List[ScaffoldResult] = []
        if beam > 1:
            outs = await asyncio.gather(*[
                self._solve_core(task, context) for _ in range(beam)])
            texts = [t for t, _r in outs]
            attempts = [r for _t, r in outs]
            votes: Dict[str, int] = {}
            for t in texts:
                a = extract_answer(t)
                votes[_norm_number(a)] = votes.get(_norm_number(a), 0) + 1
            best = max(votes.items(), key=lambda kv: kv[1])[0] if votes else ""
            merged = ScaffoldResult(
                answer=str(best),
                n_llm_calls=sum(r.n_llm_calls for r in attempts),
                tool_calls=sum(r.tool_calls for r in attempts),
                decomposed=any(r.decomposed for r in attempts),
                transcript=[{"stage": "beam_vote", "text": str(votes)}],
            )
            result = merged
        else:
            text, result = await self._solve_core(
                task, context,
                force_plan=route_deliberate and not self.g.planning.decomposition)

        # ---- verification pass ----------------------------------------
        if deliberate and result.answer != "":
            vmsgs = [
                {"role": "system", "content":
                    "You are a strict verifier. Re-derive the answer "
                    "independently. If the proposed answer is wrong, reply "
                    "with the corrected reasoning and FINAL_ANSWER. If it is "
                    "right, reply VERDICT: CORRECT and repeat FINAL_ANSWER."},
                {"role": "user", "content":
                    task.prompt + f"\n\nPROPOSED ANSWER: {result.answer}"},
            ]
            vtext, ok = await self._chat(vmsgs, temperature=0.1)
            result.n_llm_calls += 1
            result.verified = True
            va = extract_answer(vtext)
            if ok and va and _norm_number(va) != _norm_number(result.answer):
                result.self_corrected = True
                result.answer = va
                result.transcript.append({"stage": "verify_fix",
                                          "text": vtext[:500]})

        # ---- critic gate ------------------------------------------------
        if self.g.control.critic_enabled and result.answer != "":
            cmsgs = [
                {"role": "system", "content":
                    "You are a critic. Check the candidate answer against "
                    "the task requirements (units, edge cases, arithmetic). "
                    "If it fails any check, provide FINAL_ANSWER with your "
                    "fix; otherwise reply CRITIC: PASS."},
                {"role": "user", "content":
                    task.prompt + f"\n\nCANDIDATE ANSWER: {result.answer}"},
            ]
            ctext, ok = await self._chat(cmsgs, temperature=0.1)
            result.n_llm_calls += 1
            if ok and "CRITIC: PASS" not in ctext.upper():
                ca = extract_answer(ctext)
                if ca and _norm_number(ca) != _norm_number(result.answer):
                    result.self_corrected = True
                    result.answer = ca
                    result.transcript.append({"stage": "critic_fix",
                                              "text": ctext[:500]})

        # ---- retry-on-failure --------------------------------------------
        retries = 0
        while ((result.answer == "" or result.error == "backend_error")
               and self.g.control.retry_on_failure
               and retries < self.g.control.max_retries
               and self.calls_used < self.max_calls):
            retries += 1
            result.retried = True
            text2, result2 = await self._solve_core(task, context)
            result.n_llm_calls += result2.n_llm_calls
            result.tool_calls += result2.tool_calls
            if result2.answer:
                result.answer = result2.answer
                result.error = None

        result.answer = (result.answer or "").strip()
        return result


def score_answer(task: Task, answer: str) -> float:
    """Deterministic local verifier shared by all live environments."""
    if task.type == "code":
        raise ValueError("code tasks are scored by execution, not extraction")
    return 1.0 if _norm_number(answer) == _norm_number(task.truth) else -1.0
