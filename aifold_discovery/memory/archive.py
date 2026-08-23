"""AI-Fold Discovery: persistent memory of the search itself."""

import json
from pathlib import Path
from typing import Dict

from aifold_discovery.core.candidate import Population
from aifold_discovery.core.experiment import ExperimentStore


class DiscoveryArchive:
    """File-backed archive: populations, experiments, and derived discoveries.

    This is what makes AI-Fold a *scientific* system rather than a training
    run: every hypothesis, trajectory group, fitness delta, and diagnosis is
    retained and queryable.
    """

    def __init__(self, root: str = "./aifold_runs"):
        self.root = Path(root)
        (self.root / "populations").mkdir(parents=True, exist_ok=True)
        (self.root / "experiments").mkdir(parents=True, exist_ok=True)

    def save_population(self, pop: Population, tag: str) -> Path:
        p = self.root / "populations" / (tag + ".json")
        p.write_text(json.dumps(pop.to_dict(), indent=2, default=str))
        return p

    def save_experiments(self, store: ExperimentStore) -> Path:
        p = self.root / "experiments" / "all_records.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for rec in store.records.values():
                f.write(json.dumps(rec.to_dict(), default=str) + "\n")
        return p

    def save_summary(self, summary: Dict) -> Path:
        p = self.root / "discovery_summary.json"
        p.write_text(json.dumps(summary, indent=2, default=str))
        return p

    # ------------------------------------------------------------------
    @classmethod
    def persist_run(cls, engine, root: str = "./aifold_runs", tag: str = "run"):
        arch = cls(root)
        paths = {
            "population": arch.save_population(engine.population, tag),
            "experiments": arch.save_experiments(engine.store),
        }
        return arch, paths
