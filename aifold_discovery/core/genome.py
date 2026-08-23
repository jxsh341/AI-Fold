"""AI-Fold Discovery: Candidate Genome

A CandidateGenome is the complete, heritable specification of an AI system.
This is what evolution mutates and recombines; it is NOT model weights.
Weights are optimized by RL (Atropos); structure is evolved by AI-Fold.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import hashlib
import json
import uuid


@dataclass
class ModelComponent:
    """Model layer of a candidate system."""
    reasoning_model: str = "default"
    verifier_model: Optional[str] = None          # None = no self-verification
    verifier_enabled: bool = False


@dataclass
class MemoryComponent:
    """Memory architecture of a candidate system."""
    working_memory_size: int = 8
    episodic_memory: bool = False
    semantic_memory: bool = False
    retrieval_k: int = 4


@dataclass
class PlanningComponent:
    """Planning strategy of a candidate system."""
    decomposition: bool = False                   # task decomposition on/off
    search_algorithm: str = "none"                # none | bfs | dfs | mcts | beam
    search_depth: int = 1
    max_planning_tokens: int = 2048


@dataclass
class ToolComponent:
    """Tool access of a candidate system."""
    enabled_tools: List[str] = field(default_factory=list)   # e.g. ["code", "browser"]
    max_tool_calls_per_episode: int = 4


@dataclass
class ControlComponent:
    """Control/routing logic of a candidate system."""
    router_type: str = "single"                   # single | conditional | mixture
    critic_enabled: bool = False                  # internal critic that gates actions
    retry_on_failure: bool = True
    max_retries: int = 2


@dataclass
class TrainingComponent:
    """Training strategy applied to this candidate (RL-side)."""
    algorithm: str = "grpo"                       # grpo | sft | dpo | distill
    group_size: int = 16
    learning_rate: float = 1e-6
    total_steps: int = 200


@dataclass
class CandidateGenome:
    """
    Complete heritable specification of an AI system candidate.

    The genome is the unit of evolution. Two genomes are considered
    structurally identical iff their canonical hash matches.
    """
    model: ModelComponent = field(default_factory=ModelComponent)
    memory: MemoryComponent = field(default_factory=MemoryComponent)
    planning: PlanningComponent = field(default_factory=PlanningComponent)
    tools: ToolComponent = field(default_factory=ToolComponent)
    control: ControlComponent = field(default_factory=ControlComponent)
    training: TrainingComponent = field(default_factory=TrainingComponent)

    # Lineage metadata (not inherited by mutation unless copied explicitly)
    genome_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0
    mutation_history: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    def canonical_dict(self) -> Dict[str, Any]:
        """Structural fields only (excludes lineage metadata)."""
        return {
            "model": asdict(self.model),
            "memory": asdict(self.memory),
            "planning": asdict(self.planning),
            "tools": asdict(self.tools),
            "control": asdict(self.control),
            "training": asdict(self.training),
        }

    def structural_hash(self) -> str:
        """Deterministic hash over structural content."""
        blob = json.dumps(self.canonical_dict(), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def clone(self, new_id: bool = True) -> "CandidateGenome":
        """Deep copy with fresh identity (used for mutation/crossover children)."""
        child = CandidateGenome(
            model=ModelComponent(**asdict(self.model)),
            memory=MemoryComponent(**asdict(self.memory)),
            planning=PlanningComponent(**asdict(self.planning)),
            tools=ToolComponent(enabled_tools=list(self.tools.enabled_tools),
                                max_tool_calls_per_episode=self.tools.max_tool_calls_per_episode),
            control=ControlComponent(**asdict(self.control)),
            training=TrainingComponent(**asdict(self.training)),
            generation=self.generation,
        )
        if new_id:
            child.genome_id = str(uuid.uuid4())[:8]
        else:
            child.genome_id = self.genome_id
        return child

    def describe(self) -> str:
        """Human-readable one-line description."""
        parts = []
        parts.append(f"model={self.model.reasoning_model}")
        if self.model.verifier_enabled:
            parts.append("verifier")
        if self.memory.episodic_memory or self.memory.semantic_memory:
            parts.append(f"memory(w={self.memory.working_memory_size}"
                         f"{'+episodic' if self.memory.episodic_memory else ''}"
                         f"{'+semantic' if self.memory.semantic_memory else ''})")
        if self.planning.decomposition:
            parts.append(f"plan({self.planning.search_algorithm},d={self.planning.search_depth})")
        if self.tools.enabled_tools:
            parts.append(f"tools={','.join(self.tools.enabled_tools)}")
        if self.control.critic_enabled:
            parts.append("critic")
        if self.control.router_type != "single":
            parts.append(f"router={self.control.router_type}")
        return " + ".join(parts) if parts else "baseline"

    def to_dict(self) -> Dict[str, Any]:
        d = self.canonical_dict()
        d["genome_id"] = self.genome_id
        d["parent_ids"] = list(self.parent_ids)
        d["generation"] = self.generation
        d["mutation_history"] = list(self.mutation_history)
        d["description"] = self.describe()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CandidateGenome":
        g = cls(
            model=ModelComponent(**d["model"]),
            memory=MemoryComponent(**d["memory"]),
            planning=PlanningComponent(**d["planning"]),
            tools=ToolComponent(**d["tools"]),
            control=ControlComponent(**d["control"]),
            training=TrainingComponent(**d["training"]),
            genome_id=d.get("genome_id", str(uuid.uuid4())[:8]),
            parent_ids=list(d.get("parent_ids", [])),
            generation=int(d.get("generation", 0)),
            mutation_history=list(d.get("mutation_history", [])),
        )
        return g


def baseline_genome() -> CandidateGenome:
    """The A in A/B/C/D/E/F/G — plain single-model agent, no extras."""
    g = CandidateGenome()
    g.mutation_history = ["origin:baseline"]
    return g
