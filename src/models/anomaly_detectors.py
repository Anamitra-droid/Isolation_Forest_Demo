"""
Anomaly Detection Models Module

Implements multiple anomaly detection algorithms for comparison.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Base class for anomaly detectors."""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.model = None
        self.params = kwargs
    
    def fit(self, X: np.ndarray):
        """Fit the model."""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (-1 for anomaly, 1 for normal)."""
        raise NotImplementedError
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        raise NotImplementedError
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.params


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest anomaly detector."""
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: Optional[float] = None,
        max_features: float = 1.0,
        bootstrap: bool = False,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ):
        super().__init__("Isolation Forest", **kwargs)
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.params.update({
            'contamination': contamination,
            'n_estimators': n_estimators,
            'max_samples': max_samples,
            'max_features': max_features,
            'bootstrap': bootstrap,
            'random_state': random_state
        })
    
    def fit(self, X: np.ndarray):
        """Fit Isolation Forest."""
        self.model.fit(X)
        logger.info(f"Fitted {self.name} with {self.model.n_estimators} trees")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        return self.model.predict(X)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        return self.model.decision_function(X)


class EllipticEnvelopeDetector(AnomalyDetector):
    """Elliptic Envelope anomaly detector."""
    
    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__("Elliptic Envelope", **kwargs)
        self.model = EllipticEnvelope(
            contamination=contamination,
            random_state=random_state
        )
        self.params.update({
            'contamination': contamination,
            'random_state': random_state
        })
    
    def fit(self, X: np.ndarray):
        """Fit Elliptic Envelope."""
        self.model.fit(X)
        logger.info(f"Fitted {self.name}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        return self.model.predict(X)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        return self.model.decision_function(X)


class LocalOutlierFactorDetector(AnomalyDetector):
    """Local Outlier Factor anomaly detector."""
    
    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1,
        novelty: bool = True,
        **kwargs
    ):
        super().__init__("Local Outlier Factor", **kwargs)
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=novelty
        )
        self.params.update({
            'n_neighbors': n_neighbors,
            'contamination': contamination,
            'novelty': novelty
        })
    
    def fit(self, X: np.ndarray):
        """Fit LOF."""
        self.model.fit(X)
        logger.info(f"Fitted {self.name} with {self.model.n_neighbors} neighbors")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        return self.model.predict(X)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        return -self.model.score_samples(X)  # Negative because lower is more anomalous


class OneClassSVMDetector(AnomalyDetector):
    """One-Class SVM anomaly detector."""
    
    def __init__(
        self,
        kernel: str = 'rbf',
        nu: float = 0.1,
        gamma: str = 'scale',
        **kwargs
    ):
        super().__init__("One-Class SVM", **kwargs)
        self.model = OneClassSVM(
            kernel=kernel,
            nu=nu,
            gamma=gamma
        )
        self.params.update({
            'kernel': kernel,
            'nu': nu,
            'gamma': gamma
        })
    
    def fit(self, X: np.ndarray):
        """Fit One-Class SVM."""
        self.model.fit(X)
        logger.info(f"Fitted {self.name} with {self.model.kernel} kernel")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        return self.model.predict(X)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        return self.model.decision_function(X)


class DBSCANDetector(AnomalyDetector):
    """DBSCAN-based anomaly detector."""
    
    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        **kwargs
    ):
        super().__init__("DBSCAN", **kwargs)
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.params.update({
            'eps': eps,
            'min_samples': min_samples
        })
    
    def fit(self, X: np.ndarray):
        """Fit DBSCAN."""
        self.model.fit(X)
        logger.info(f"Fitted {self.name} with eps={self.model.eps}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        labels = self.model.fit_predict(X)
        # DBSCAN: -1 is noise/anomaly, others are clusters
        predictions = np.where(labels == -1, -1, 1)
        return predictions
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores (distance to nearest core point)."""
        # For DBSCAN, we use cluster labels as scores
        labels = self.model.fit_predict(X)
        scores = np.where(labels == -1, -1.0, 1.0)
        return scores


def create_detector(
    detector_type: str,
    **kwargs
) -> AnomalyDetector:
    """
    Factory function to create anomaly detectors.
    
    Args:
        detector_type: Type of detector ('isolation_forest', 'elliptic_envelope',
                      'lof', 'one_class_svm', 'dbscan')
        **kwargs: Detector-specific parameters
        
    Returns:
        AnomalyDetector instance
    """
    detectors = {
        'isolation_forest': IsolationForestDetector,
        'elliptic_envelope': EllipticEnvelopeDetector,
        'lof': LocalOutlierFactorDetector,
        'one_class_svm': OneClassSVMDetector,
        'dbscan': DBSCANDetector
    }
    
    if detector_type not in detectors:
        raise ValueError(
            f"Unknown detector type: {detector_type}. "
            f"Available: {list(detectors.keys())}"
        )
    
    return detectors[detector_type](**kwargs)
