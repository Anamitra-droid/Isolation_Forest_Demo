"""
Data Loading and Generation Module

Handles data loading, generation, and preprocessing for anomaly detection.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)


class DataGenerator:
    """Generate synthetic datasets for anomaly detection."""
    
    @staticmethod
    def generate_blob_data(
        n_samples: int = 1000,
        n_features: int = 2,
        contamination: float = 0.1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate blob-shaped data with anomalies.
        
        Args:
            n_samples: Total number of samples
            n_features: Number of features
            contamination: Proportion of outliers
            random_state: Random seed
            
        Returns:
            Tuple of (X, y) where y is 1 for normal, -1 for anomalies
        """
        np.random.seed(random_state)
        n_normal = int(n_samples * (1 - contamination))
        n_anomalies = n_samples - n_normal
        
        # Generate normal data
        X_normal, _ = make_blobs(
            n_samples=n_normal,
            centers=[[0, 0]],
            cluster_std=1.0,
            n_features=n_features,
            random_state=random_state
        )
        
        # Generate anomalies
        X_anomalies = np.random.uniform(
            low=-6, high=6, size=(n_anomalies, n_features)
        )
        
        X = np.vstack([X_normal, X_anomalies])
        y = np.concatenate([np.ones(n_normal), -np.ones(n_anomalies)])
        
        # Shuffle
        indices = np.random.permutation(n_samples)
        return X[indices], y[indices]
    
    @staticmethod
    def generate_moon_data(
        n_samples: int = 1000,
        contamination: float = 0.1,
        noise: float = 0.1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate moon-shaped data with anomalies."""
        np.random.seed(random_state)
        n_normal = int(n_samples * (1 - contamination))
        n_anomalies = n_samples - n_normal
        
        X_normal, _ = make_moons(
            n_samples=n_normal,
            noise=noise,
            random_state=random_state
        )
        
        X_anomalies = np.random.uniform(
            low=-2, high=3, size=(n_anomalies, 2)
        )
        
        X = np.vstack([X_normal, X_anomalies])
        y = np.concatenate([np.ones(n_normal), -np.ones(n_anomalies)])
        
        indices = np.random.permutation(n_samples)
        return X[indices], y[indices]
    
    @staticmethod
    def generate_circle_data(
        n_samples: int = 1000,
        contamination: float = 0.1,
        noise: float = 0.1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate circle-shaped data with anomalies."""
        np.random.seed(random_state)
        n_normal = int(n_samples * (1 - contamination))
        n_anomalies = n_samples - n_normal
        
        X_normal, _ = make_circles(
            n_samples=n_normal,
            noise=noise,
            factor=0.5,
            random_state=random_state
        )
        
        X_anomalies = np.random.uniform(
            low=-1.5, high=1.5, size=(n_anomalies, 2)
        )
        
        X = np.vstack([X_normal, X_anomalies])
        y = np.concatenate([np.ones(n_normal), -np.ones(n_anomalies)])
        
        indices = np.random.permutation(n_samples)
        return X[indices], y[indices]
    
    @staticmethod
    def generate_high_dimensional_data(
        n_samples: int = 1000,
        n_features: int = 20,
        contamination: float = 0.1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate high-dimensional data with anomalies."""
        np.random.seed(random_state)
        n_normal = int(n_samples * (1 - contamination))
        n_anomalies = n_samples - n_normal
        
        # Normal data with correlation
        mean = np.zeros(n_features)
        cov = np.eye(n_features)
        X_normal = np.random.multivariate_normal(mean, cov, n_normal)
        
        # Anomalies with different distribution
        X_anomalies = np.random.multivariate_normal(
            mean + 3, cov * 2, n_anomalies
        )
        
        X = np.vstack([X_normal, X_anomalies])
        y = np.concatenate([np.ones(n_normal), -np.ones(n_anomalies)])
        
        indices = np.random.permutation(n_samples)
        return X[indices], y[indices]


class DataPreprocessor:
    """Preprocess data for anomaly detection."""
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: Type of scaler ('standard' or 'robust')
        """
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler and transform data."""
        return self.scaler.fit_transform(X)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler."""
        return self.scaler.transform(X)
    
    def fit(self, X: np.ndarray):
        """Fit scaler on data."""
        self.scaler.fit(X)


def load_data(
    dataset_type: str = 'blob',
    n_samples: int = 1000,
    n_features: int = 2,
    contamination: float = 0.1,
    normalize: bool = True,
    scaler_type: str = 'standard',
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load or generate dataset for anomaly detection.
    
    Args:
        dataset_type: Type of dataset ('blob', 'moon', 'circle', 'high_dim')
        n_samples: Number of samples
        n_features: Number of features (for blob and high_dim)
        contamination: Proportion of anomalies
        normalize: Whether to normalize data
        scaler_type: Type of scaler ('standard' or 'robust')
        random_state: Random seed
        
    Returns:
        Tuple of (X, y) where y is 1 for normal, -1 for anomalies
    """
    generator = DataGenerator()
    
    if dataset_type == 'blob':
        X, y = generator.generate_blob_data(
            n_samples, n_features, contamination, random_state
        )
    elif dataset_type == 'moon':
        X, y = generator.generate_moon_data(
            n_samples, contamination, random_state=random_state
        )
    elif dataset_type == 'circle':
        X, y = generator.generate_circle_data(
            n_samples, contamination, random_state=random_state
        )
    elif dataset_type == 'high_dim':
        X, y = generator.generate_high_dimensional_data(
            n_samples, n_features, contamination, random_state
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    if normalize:
        preprocessor = DataPreprocessor(scaler_type)
        X = preprocessor.fit_transform(X)
        logger.info(f"Data normalized using {scaler_type} scaler")
    
    logger.info(f"Generated {dataset_type} dataset: {X.shape[0]} samples, "
                f"{X.shape[1]} features, {np.sum(y == -1)} anomalies")
    
    return X, y
