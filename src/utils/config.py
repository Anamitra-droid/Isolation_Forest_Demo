"""
Configuration Management Module

Handle configuration loading and validation.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration."""
    detector_type: str = 'isolation_forest'
    contamination: float = 0.1
    n_estimators: int = 100
    random_state: int = 42
    n_jobs: int = -1
    
    # Isolation Forest specific
    max_samples: Optional[float] = None
    max_features: float = 1.0
    bootstrap: bool = False
    
    # LOF specific
    n_neighbors: int = 20
    
    # One-Class SVM specific
    kernel: str = 'rbf'
    nu: float = 0.1
    gamma: str = 'scale'
    
    # DBSCAN specific
    eps: float = 0.5
    min_samples: int = 5


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_type: str = 'blob'
    n_samples: int = 1000
    n_features: int = 2
    contamination: float = 0.1
    normalize: bool = True
    scaler_type: str = 'standard'
    random_state: int = 42


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    data: DataConfig
    model: ModelConfig
    evaluation: Dict[str, Any]
    visualization: Dict[str, Any]
    output_dir: str = 'outputs'
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            evaluation=config_dict.get('evaluation', {}),
            visualization=config_dict.get('visualization', {}),
            output_dir=config_dict.get('output_dir', 'outputs')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'evaluation': self.evaluation,
            'visualization': self.visualization,
            'output_dir': self.output_dir
        }
    
    def save(self, path: str):
        """Save config to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        logger.info(f"Config saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load config from file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return cls.from_dict(config_dict)


def get_default_config() -> ExperimentConfig:
    """Get default configuration."""
    return ExperimentConfig(
        data=DataConfig(),
        model=ModelConfig(),
        evaluation={
            'calculate_roc': True,
            'calculate_pr': True,
            'save_metrics': True
        },
        visualization={
            'plot_results': True,
            'plot_roc': True,
            'plot_pr': True,
            'plot_comparison': False,
            'save_plots': True
        }
    )
