"""AI-Fold v0.1 - An AlphaFold-inspired system for AI trajectory prediction"""

__version__ = "0.1.0"

from aifold.config import (
    ModelConfig,
    EntityTypeConfig,
    RelationTypeConfig,
    ExperimentConfig,
    EXPERIMENTS,
    DEFAULT_MODEL_CONFIG,
)

from aifold.model import AIModel, create_model_from_experiment

__all__ = [
    'ModelConfig',
    'EntityTypeConfig', 
    'RelationTypeConfig',
    'ExperimentConfig',
    'EXPERIMENTS',
    'DEFAULT_MODEL_CONFIG',
    'AIModel',
    'create_model_from_experiment',
]