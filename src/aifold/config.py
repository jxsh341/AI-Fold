"""AI-Fold v0.1 Configuration"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    """Main model configuration matching the v0.1 architecture spec."""
    
    # Entity representation
    d_H: int = 384              # Entity state dimension
    d_P: int = 128              # Pair/relation state dimension
    d_Z: int = 512              # Latent state dimension
    
    # Relational trunk
    num_relational_blocks: int = 8
    num_heads: int = 16
    
    # Recycling
    num_recycles: int = 4
    
    # Latent diffusion
    num_diffusion_blocks: int = 12
    num_diffusion_heads: int = 16
    num_diffusion_steps: int = 64       # Increased from 32 for better quality
    sigma_data: float = 1.0
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    num_samples: int = 8                # M candidates
    
    # Confidence head
    num_confidence_blocks: int = 4
    
    # Trajectory
    horizon_T: int = 8                  # Future steps
    step_duration: float = 1.0          # Fixed duration per step
    
    # Action space
    num_action_classes: int = 23        # Matches RelationTypeConfig.NUM_TYPES
    
    # Training
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    
    # Ablation flags
    use_trirel: bool = False            # Experiment H
    use_retrieval: bool = False         # Experiment J
    use_confidence_recycling: bool = False  # Experiment G/I


@dataclass
class EntityTypeConfig:
    """Entity type definitions for AI-Fold."""
    
    # Core entity types (16 types as specified)
    ENTITY_TYPES = [
        "AGENT",
        "MODEL", 
        "TASK",
        "GOAL",
        "OBSERVATION",
        "ACTION",
        "TOOL",
        "MEMORY",
        "DOCUMENT",
        "RESOURCE",
        "CONSTRAINT",
        "ENVIRONMENT",
        "EVENT",
        "LATENT_STATE",
        "COMPUTATION",
        "RETRIEVED_TRAJECTORY",
    ]
    
    @classmethod
    def get_type_id(cls, name: str) -> int:
        return cls.ENTITY_TYPES.index(name)
    
    @classmethod
    def get_type_name(cls, idx: int) -> str:
        return cls.ENTITY_TYPES[idx]
    
    NUM_TYPES = len(ENTITY_TYPES)


@dataclass
class RelationTypeConfig:
    """Relation type definitions for AI-Fold (24 types as specified)."""
    
    RELATION_TYPES = [
        # Semantic (10)
        "uses",
        "depends_on",
        "causes",
        "contains",
        "retrieves",
        "supports",
        "contradicts",
        "achieves",
        "observes",
        "produces",
        # Temporal (3)
        "same_step",
        "previous_step",
        "next_step",
        # Causal (4)
        "caused_by",
        "enables",
        "precondition_for",
        "result_of",
        # Structural (5)
        "same_goal",
        "same_task",
        "same_agent",
        "same_episode",
        "same_tool",
        # No relation (1)
        "none",
    ]
    
    @classmethod
    def get_relation_id(cls, name: str) -> int:
        return cls.RELATION_TYPES.index(name)
    
    @classmethod
    def get_relation_name(cls, idx: int) -> str:
        return cls.RELATION_TYPES[idx]
    
    NUM_TYPES = len(RELATION_TYPES)


@dataclass
class ExperimentConfig:
    """Experiment configuration for the A→J ablation sequence."""
    
    # Core experiments A-G
    experiment: Literal[
        "A_flat_transformer",
        "B_entity_only", 
        "C_entity_pair",
        "D_recycling",
        "E_diffusion",
        "F_confidence",
        "G_confidence_recycle",
        "H_trirel",
        "I_adaptive_recycling",
        "J_retrieval",
    ] = "C_entity_pair"
    
    # Override specific model configs per experiment
    def get_model_config(self) -> ModelConfig:
        cfg = ModelConfig()
        
        if self.experiment == "A_flat_transformer":
            cfg.num_relational_blocks = 0
            cfg.num_recycles = 0
        elif self.experiment == "B_entity_only":
            cfg.num_relational_blocks = 8
            # No pair construction - entity-only path
        elif self.experiment == "C_entity_pair":
            pass  # Default config
        elif self.experiment == "D_recycling":
            cfg.num_recycles = 4
        elif self.experiment == "E_diffusion":
            cfg.num_diffusion_steps = 32
        elif self.experiment == "F_confidence":
            cfg.num_confidence_blocks = 4
        elif self.experiment == "G_confidence_recycle":
            cfg.use_confidence_recycling = True
        elif self.experiment == "H_trirel":
            cfg.use_trirel = True
        elif self.experiment == "I_adaptive_recycling":
            cfg.use_confidence_recycling = True
        elif self.experiment == "J_retrieval":
            cfg.use_retrieval = True
            
        return cfg


# Default configs for different experiment stages
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()

# Experiment aliases
EXPERIMENTS = {
    'A': 'A_flat_transformer',
    'B': 'B_entity_only',
    'C': 'C_entity_pair',
    'D': 'D_recycling',
    'E': 'E_diffusion',
    'F': 'F_confidence',
    'G': 'G_confidence_recycle',
    'H': 'H_trirel',
    'I': 'I_adaptive_recycling',
    'J': 'J_retrieval',
}