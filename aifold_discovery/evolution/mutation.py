"""AI-Fold Discovery: Mutation Operators.

Mutations operate on the genome (system structure), NOT on weights.
Weights are optimized by RL; structure is evolved here.

Mutation policy is fitness-aware: prefer mutations that target the
candidate's weakest measured axis. This is what makes AI-Fold's search
scientific rather than random — every mutation is a hypothesis:
"adding X should improve capability Y".
"""

import random
from typing import List, Optional, Tuple

from aifold_discovery.core.genome import (
    CandidateGenome, ModelComponent, MemoryComponent,
    PlanningComponent, ToolComponent, ControlComponent,
)
from aifold_discovery.core.fitness import AXES


# Registry of mutations: (name, axis_targeted, applicability, apply)
# axis_targeted expresses the hypothesis: this mutation targets that axis.

def _m_add_verifier(g: CandidateGenome) -> str:
    if g.model.verifier_enabled:
        return None
    g.model.verifier_enabled = True
    return "add_verifier"

def _m_add_episodic_memory(g: CandidateGenome) -> str:
    if g.memory.episodic_memory:
        return None
    g.memory.episodic_memory = True
    g.memory.retrieval_k = max(4, g.memory.retrieval_k)
    return "add_episodic_memory"

def _m_add_semantic_memory(g: CandidateGenome) -> str:
    if g.memory.semantic_memory:
        return None
    g.memory.semantic_memory = True
    return "add_semantic_memory"

def _m_expand_working_memory(g: CandidateGenome) -> str:
    old = g.memory.working_memory_size
    g.memory.working_memory_size = min(64, old * 2)
    return f"working_memory_{old}->{g.memory.working_memory_size}"

def _m_enable_decomposition(g: CandidateGenome) -> str:
    if g.planning.decomposition:
        return None
    g.planning.decomposition = True
    return "enable_decomposition"

def _m_upgrade_search(g: CandidateGenome) -> str:
    ladder = ["none", "bfs", "beam", "mcts"]
    try:
        i = ladder.index(g.planning.search_algorithm)
    except ValueError:
        i = 0
    if i >= len(ladder) - 1:
        return None
    old = g.planning.search_algorithm
    g.planning.search_algorithm = ladder[i + 1]
    return f"search_{old}->{g.planning.search_algorithm}"

def _m_deepen_search(g: CandidateGenome) -> str:
    if not g.planning.decomposition:
        g.planning.decomposition = True
    old = g.planning.search_depth
    g.planning.search_depth = min(8, old + 2)
    return f"search_depth_{old}->{g.planning.search_depth}"

def _m_add_tool_code(g: CandidateGenome) -> str:
    if "code" in g.tools.enabled_tools:
        return None
    g.tools.enabled_tools.append("code")
    return "add_tool_code"

def _m_add_tool_browser(g: CandidateGenome) -> str:
    if "browser" in g.tools.enabled_tools:
        return None
    g.tools.enabled_tools.append("browser")
    return "add_tool_browser"

def _m_raise_tool_budget(g: CandidateGenome) -> str:
    old = g.tools.max_tool_calls_per_episode
    g.tools.max_tool_calls_per_episode = min(32, old * 2)
    return f"tool_budget_{old}->{g.tools.max_tool_calls_per_episode}"

def _m_enable_critic(g: CandidateGenome) -> str:
    if g.control.critic_enabled:
        return None
    g.control.critic_enabled = True
    return "enable_critic"

def _m_conditional_router(g: CandidateGenome) -> str:
    if g.control.router_type != "single":
        return None
    g.control.router_type = "conditional"
    return "conditional_router"

def _m_enable_retries(g: CandidateGenome) -> str:
    if g.control.retry_on_failure and g.control.max_retries >= 3:
        return None
    g.control.retry_on_failure = True
    g.control.max_retries += 1
    return f"retries->{g.control.max_retries}"

def _m_bigger_rl_budget(g: CandidateGenome) -> str:
    old = g.training.total_steps
    g.training.total_steps = int(old * 1.5)
    return f"rl_steps_{old}->{g.training.total_steps}"


MUTATIONS = [
    # (name, axis_targeted, mode, apply)
    # mode: "live"   -> changes inference behavior of the scaffold
    #       "rl"     -> only meaningful with a real weight trainer
    ("add_verifier",           "self_correction", "live", _m_add_verifier),
    ("add_episodic_memory",    "memory",          "live", _m_add_episodic_memory),
    ("add_semantic_memory",    "generalization",  "live", _m_add_semantic_memory),
    ("expand_working_memory",  "memory",          "live", _m_expand_working_memory),
    ("enable_decomposition",   "planning",        "live", _m_enable_decomposition),
    ("upgrade_search",         "planning",        "live", _m_upgrade_search),
    ("deepen_search",          "planning",        "live", _m_deepen_search),
    ("add_tool_code",          "coding",          "live", _m_add_tool_code),
    ("add_tool_browser",       "tool_use",        "live", _m_add_tool_browser),
    ("raise_tool_budget",      "tool_use",        "live", _m_raise_tool_budget),
    ("enable_critic",          "self_correction", "live", _m_enable_critic),
    ("conditional_router",     "efficiency",      "live", _m_conditional_router),
    ("enable_retries",         "robustness",      "live", _m_enable_retries),
    ("bigger_rl_budget",       "reasoning",       "rl",   _m_bigger_rl_budget),
]

MUTATION_INDEX = {
    name: (axis, mode, fn) for name, axis, mode, fn in MUTATIONS
}


class Mutator:
    """Fitness-aware mutation operator.

    allow_rl=False quarantines mutations that are phenotypically silent
    without a weight trainer — otherwise 'improvements' can be credited
    to genes that changed nothing at inference time.
    """

    def __init__(self, rng: Optional[random.Random] = None,
                 targeted_ratio: float = 0.7):
        self.rng = rng or random.Random()
        self.targeted_ratio = targeted_ratio

    def applicable(self, genome: CandidateGenome,
                   allow_rl: bool = False) -> List[str]:
        """Mutations that would actually change this genome."""
        out = []
        for name, _axis, mode, fn in MUTATIONS:
            if mode == "rl" and not allow_rl:
                continue
            probe = genome.clone(new_id=False)
            test = probe.clone(new_id=False)
            if fn(test) is not None:
                out.append(name)
        del probe
        return out

    def mutate(self, parent: CandidateGenome,
               bottleneck_axis: Optional[str] = None,
               allow_rl: bool = False,
               avoid: Optional[str] = None) -> Tuple[CandidateGenome, str]:
        """Produce one mutated child. Returns (child_genome, mutation_name).

        Two anti-lock-in rules learned from live sweeps:
          - targeted fallback: a singleton bottleneck pool (only ONE mutation
            tagged with the weakest axis) would otherwise be chosen 70% of
            the time forever -> require >=2 options to use targeting;
          - avoid: never resample the mutation that produced this very parent
            (breaks router-chains where one gene replicates down a lineage).
        """
        options = self.applicable(parent, allow_rl=allow_rl)
        if avoid and avoid in options and len(options) > 1:
            options = [o for o in options if o != avoid]
        if not options:
            child = parent.clone()
            child.generation = parent.generation + 1
            child.parent_ids = [parent.genome_id]
            child.mutation_history = list(parent.mutation_history) + ["no_op"]
            return child, "no_op"

        # Targeted choice: pick mutations aligned with weakest axis.
        targeted_pool = []
        if bottleneck_axis:
            for name in options:
                ax, _mode, _fn = MUTATION_INDEX[name]
                if ax == bottleneck_axis:
                    targeted_pool.append(name)

        use_targeted = len(targeted_pool) >= 2 or (
            len(targeted_pool) == 1 and self.rng.random() < self.targeted_ratio * 0.5
        )
        pool = targeted_pool if use_targeted else options
        name = self.rng.choice(pool)
        _axis, _mode, fn = MUTATION_INDEX[name]

        child = parent.clone()
        applied = fn(child)
        applied = applied or name
        child.parent_ids = [parent.genome_id]
        child.generation = parent.generation + 1
        child.mutation_history = list(parent.mutation_history) + [applied]
        # Note: measurements are reset on the wrapping Candidate (fresh
        # FitnessVector at Candidate creation), not on the genome itself.
        return child, applied


class Crossover:
    """Recombination of two parents' genomes, per-component majority/viable mix."""

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def crossover(self, a: CandidateGenome, b: CandidateGenome) -> Tuple[CandidateGenome, str]:
        child = a.clone()
        parts = []

        # Per-component: randomly inherit from either parent, but only take
        # additions (never regress a feature both parents agree is useless).
        if self.rng.random() < 0.5 and b.model.verifier_enabled:
            child.model.verifier_enabled = True
            parts.append("verifier<-" + b.genome_id)
        if self.rng.random() < 0.5 and (b.memory.episodic_memory and not child.memory.episodic_memory):
            child.memory.episodic_memory = True
            parts.append("episodic<-" + b.genome_id)
        if self.rng.random() < 0.5 and (b.memory.semantic_memory and not child.memory.semantic_memory):
            child.memory.semantic_memory = True
            parts.append("semantic<-" + b.genome_id)
        if self.rng.random() < 0.5 and (b.planning.decomposition and not child.planning.decomposition):
            child.planning.decomposition = True
            child.planning.search_algorithm = b.planning.search_algorithm
            parts.append("planner<-" + b.genome_id)
        for tool in b.tools.enabled_tools:
            if tool not in child.tools.enabled_tools and self.rng.random() < 0.5:
                child.tools.enabled_tools.append(tool)
                parts.append(f"tool:{tool}<-{b.genome_id}")
        if self.rng.random() < 0.5 and (b.control.critic_enabled and not child.control.critic_enabled):
            child.control.critic_enabled = True
            parts.append("critic<-" + b.genome_id)
        # Take the stronger RL budget
        if b.training.total_steps > child.training.total_steps:
            child.training.total_steps = b.training.total_steps
            parts.append("rl_budget")

        child.parent_ids = [a.genome_id, b.genome_id]
        child.generation = max(a.generation, b.generation) + 1
        child.mutation_history = list(a.mutation_history)
        label = "crossover[" + ",".join(parts) + "]" if parts else "crossover[clone_a]"
        child.mutation_history.append(label)
        return child, label
