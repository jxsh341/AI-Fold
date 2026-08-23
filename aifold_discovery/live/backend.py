"""AI-Fold Live: LLM backends.

Pluggable inference backends behind one interface. Anything
OpenAI-compatible works out of the box: OpenAI, Together, Groq,
DeepSeek, OpenRouter, vLLM (`vllm serve`), SGLang, llama.cpp server,
LM Studio, Ollama (`ollama serve` exposes /v1).

Selection order:
    1. AIFOLD_BASE_URL (+ optional AIFOLD_API_KEY, AIFOLD_MODEL)   explicit
    2. OPENAI_API_KEY            -> api.openai.com
    3. GROQ_API_KEY              -> groq compatible endpoint
    4. http://localhost:11434/v1 -> ollama (probed)
    5. MockBackend               -> deterministic offline verification

Zero hard dependencies: HTTP via urllib in worker threads.
"""

import asyncio
import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    backend: str = ""
    error: Optional[str] = None


class LLMBackend:
    name = "base"

    async def chat(self, messages: List[Dict[str, str]],
                   temperature: float = 0.7,
                   max_tokens: int = 1024,
                   stop: Optional[List[str]] = None) -> ChatResult:
        raise NotImplementedError

    def usage_summary(self) -> Dict[str, float]:
        return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def _post_json(url: str, payload: Dict[str, Any],
               headers: Optional[Dict[str, str]] = None,
               timeout: float = 120.0) -> Dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class OpenAICompatBackend(LLMBackend):
    """Works with any /v1/chat/completions endpoint."""

    def __init__(self, base_url: str, api_key: str = "x", model: str = "default",
                 max_concurrency: int = 4):
        self.name = f"openai-compat:{base_url}"
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._sem = asyncio.Semaphore(max_concurrency)
        self._stats = {"calls": 0, "errors": 0, "prompt_tokens": 0,
                       "completion_tokens": 0}

    async def chat(self, messages, temperature=0.7, max_tokens=1024, stop=None):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop
        headers = {}
        if self.api_key and self.api_key != "x":
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with self._sem:
            t0 = time.time()
            try:
                data = await asyncio.to_thread(
                    _post_json, f"{self.base_url}/chat/completions",
                    payload, headers,
                )
                self._stats["calls"] += 1
                choice = data["choices"][0]["message"]["content"] or ""
                usage = data.get("usage", {})
                self._stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
                self._stats["completion_tokens"] += usage.get("completion_tokens", 0)
                return ChatResult(
                    text=choice,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    latency_s=time.time() - t0,
                    backend=self.name,
                )
            except Exception as e:
                self._stats["errors"] += 1
                return ChatResult(text="", latency_s=time.time() - t0,
                                  backend=self.name, error=str(e))

    def usage_summary(self):
        return dict(self._stats)


class MockBackend(LLMBackend):
    """Deterministic offline backend for verifying the live code path.

    Produces plausible agent-style responses whose QUALITY responds to hints
    embedded in the system/user messages, so scaffolding differences
    (verifier pass, decomposition, retries) measurably change outcomes —
    exactly like a real model would respond differently to better
    scaffolding. Never used when a real backend is detected.
    """

    name = "mock"

    def __init__(self, base_quality: float = 0.5, seed: int = 0):
        import random
        self.rng = random.Random(seed)
        self.base_quality = base_quality
        self._stats = {"calls": 0, "errors": 0, "prompt_tokens": 0,
                       "completion_tokens": 0}
        self._last_hint_quality: Optional[float] = None

    def _hint_quality(self, messages) -> float:
        """Mock quality rises with scaffolding signals in the prompt."""
        blob = " ".join(m.get("content", "") for m in messages).lower()
        q = self.base_quality
        q += 0.10 if "step" in blob and "plan" in blob else 0.0     # decomposition
        q += 0.08 if "verify" in blob or "check" in blob else 0.0   # verifier
        q += 0.06 if "critique" in blob else 0.0                    # critic
        q += 0.05 if "relevant context" in blob or "memory" in blob else 0.0
        return min(0.95, q)

    async def chat(self, messages, temperature=0.7, max_tokens=1024, stop=None):
        await asyncio.sleep(0)  # yield; keeps async semantics identical
        self._stats["calls"] += 1
        q = self._hint_quality(messages)
        user = next((m["content"] for m in reversed(messages)
                     if m.get("role") == "user"), "")
        sys = next((m["content"] for m in messages if m.get("role") == "system"), "")

        # Task-aware response generation -------------------------------
        if "FINAL_ANSWER:" in (sys + user) or "answer" in sys.lower():
            correct_prob = q
            ok = self.rng.random() < correct_prob
            # Extract expected answer hint from task block when present
            ans = self._solve_or_fail(user, ok)
            return ChatResult(text=ans,
                              prompt_tokens=len(user) // 4,
                              completion_tokens=64,
                              backend=self.name)

        # Default: echo-style reasoning text
        return ChatResult(text="working through the problem...",
                          prompt_tokens=len(user) // 4,
                          completion_tokens=32, backend=self.name)

    @staticmethod
    def _extract_task(user: str) -> Optional[Dict]:
        try:
            start = user.index("{")
            end = user.rindex("}") + 1
            return json.loads(user[start:end])
        except Exception:
            return None

    def _solve_or_fail(self, user: str, ok: bool) -> str:
        task = self._extract_task(user)
        if not task:
            return "FINAL_ANSWER: unknown"
        ttype = task.get("type", "")
        truth = task.get("truth")
        if ttype == "math":
            return f"FINAL_ANSWER: {truth}" if ok else \
                   f"FINAL_ANSWER: {self._wrong_numeric(truth)}"
        if ttype == "code":
            return task.get("solution_template", "def stub(): pass") if ok \
                else task.get("broken_solution", "def stub(): pass")
        if ttype == "memory":
            return f"FINAL_ANSWER: {truth}" if ok else "FINAL_ANSWER: unknown"
        if ttype == "selfcorrection":
            if task.get("stage") == "verify":
                # verification stage catches the trap with prob tied to ok
                caught = ok
                return "VERDICT: INCORRECT" if caught else "VERDICT: CORRECT"
            return f"FINAL_ANSWER: {truth}" if ok else \
                   f"FINAL_ANSWER: {task.get('trap', '0')}"
        return f"FINAL_ANSWER: {truth}" if ok and truth is not None else "FINAL_ANSWER: unknown"

    @staticmethod
    def _wrong_numeric(truth) -> str:
        """Deterministic wrong answer (offset +3) for offline verification."""
        try:
            return str(int(float(truth)) + 3)
        except (TypeError, ValueError):
            return "0"

    def usage_summary(self):
        return dict(self._stats)


# ----------------------------------------------------------------------
# Detection


async def _probe(url: str, headers: Optional[Dict] = None,
                 timeout: float = 2.0) -> bool:
    def _get():
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    try:
        return await asyncio.to_thread(_get)
    except Exception:
        return False


async def detect_backend(override_base: Optional[str] = None,
                         override_key: Optional[str] = None,
                         override_model: Optional[str] = None,
                         allow_mock: bool = True,
                         seed: int = 0) -> LLMBackend:
    """Find the best available backend; never raises unless allow_mock=False."""

    base = override_base or os.environ.get("AIFOLD_BASE_URL")
    key = override_key or os.environ.get("AIFOLD_API_KEY",
                                         os.environ.get("OPENAI_API_KEY"))
    model = override_model or os.environ.get("AIFOLD_MODEL")

    if base:
        m = model or ("gpt-4o-mini" if "openai.com" in base else "default")
        k = key or ("x" if "localhost" in base or "127.0.0.1" in base else "")
        return OpenAICompatBackend(base, api_key=k, model=m)

    if os.environ.get("OPENAI_API_KEY"):
        return OpenAICompatBackend(
            "https://api.openai.com/v1",
            api_key=os.environ["OPENAI_API_KEY"],
            model=model or "gpt-4o-mini",
        )

    nv = os.environ.get("NVAPI_KEY") or os.environ.get("NVIDIA_API_KEY")
    if nv or (key or "").startswith("nvapi-"):
        return OpenAICompatBackend(
            base_url=os.environ.get(
                "AIFOLD_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            api_key=nv or key,
            model=model or os.environ.get(
                "AIFOLD_MODEL", "meta/llama-3.1-8b-instruct"),
        )

    if os.environ.get("GROQ_API_KEY"):
        return OpenAICompatBackend(
            "https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
            model=model or "llama-3.1-8b-instant",
        )

    # Local probes
    ollama_headers = {}
    if os.environ.get("OLLAMA_API_KEY"):
        ollama_headers["Authorization"] = f"Bearer {os.environ['OLLAMA_API_KEY']}"
    if await _probe("http://localhost:11434/v1/models"):
        return OpenAICompatBackend("http://localhost:11434/v1",
                                   api_key=os.environ.get("OLLAMA_API_KEY", "x"),
                                   model=model or os.environ.get(
                                       "OLLAMA_MODEL", "llama3.1"))
    for port in (8000, 1234, 5000, 8080):
        if await _probe(f"http://localhost:{port}/v1/models"):
            return OpenAICompatBackend(f"http://localhost:{port}/v1",
                                       api_key=key or "x", model=model or "default")

    if allow_mock:
        return MockBackend(seed=seed)
    raise RuntimeError(
        "No LLM backend found. Set AIFOLD_BASE_URL/AIFOLD_API_KEY/AIFOLD_MODEL, "
        "or OPENAI_API_KEY / GROQ_API_KEY, or start Ollama on :11434."
    )
