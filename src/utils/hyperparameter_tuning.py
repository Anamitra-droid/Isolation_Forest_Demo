"""
Hyperparameter Tuning Module

Grid search and cross-validation for anomaly detection models.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import cross_val_score
import logging

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Hyperparameter tuning for anomaly detection models."""
    
    def __init__(self, cv: int = 5, scoring: str = 'roc_auc'):
        """
        Initialize tuner.
        
        Args:
            cv: Number of cross-validation folds
            scoring: Scoring metric
        """
        self.cv = cv
        self.scoring = scoring
    
    def grid_search(
        self,
        model_class,
        param_grid: Dict[str, List[Any]],
        X: np.ndarray,
        y: np.ndarray,
        n_jobs: int = -1
    ) -> Tuple[Dict[str, Any], float]:
        """
        Perform grid search for hyperparameters.
        
        Args:
            model_class: Model class to instantiate
            param_grid: Parameter grid dictionary
            X: Feature matrix
            y: Labels
            n_jobs: Number of parallel jobs
            
        Returns:
            Tuple of (best_params, best_score)
        """
        best_score = -np.inf
        best_params = None
        
        param_combinations = list(ParameterGrid(param_grid))
        logger.info(f"Testing {len(param_combinations)} parameter combinations...")
        
        for idx, params in enumerate(param_combinations):
            try:
                model = model_class(**params)
                model.fit(X)
                y_pred = model.predict(X)
                
                # Calculate score (using F1 for anomaly detection)
                from sklearn.metrics import f1_score
                y_true_binary = (y == -1).astype(int)
                y_pred_binary = (y_pred == -1).astype(int)
                score = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"Completed {idx + 1}/{len(param_combinations)} combinations")
            
            except Exception as e:
                logger.warning(f"Failed with params {params}: {e}")
                continue
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {best_score:.4f}")
        
        return best_params, best_score
    
    def random_search(
        self,
        model_class,
        param_distributions: Dict[str, List[Any]],
        X: np.ndarray,
        y: np.ndarray,
        n_iter: int = 20,
        random_state: int = 42
    ) -> Tuple[Dict[str, Any], float]:
        """
        Perform random search for hyperparameters.
        
        Args:
            model_class: Model class to instantiate
            param_distributions: Parameter distributions dictionary
            X: Feature matrix
            y: Labels
            n_iter: Number of iterations
            random_state: Random seed
            
        Returns:
            Tuple of (best_params, best_score)
        """
        np.random.seed(random_state)
        best_score = -np.inf
        best_params = None
        
        logger.info(f"Testing {n_iter} random parameter combinations...")
        
        for idx in range(n_iter):
            # Sample random parameters
            params = {}
            for param_name, param_values in param_distributions.items():
                params[param_name] = np.random.choice(param_values)
            
            try:
                model = model_class(**params)
                model.fit(X)
                y_pred = model.predict(X)
                
                from sklearn.metrics import f1_score
                y_true_binary = (y == -1).astype(int)
                y_pred_binary = (y_pred == -1).astype(int)
                score = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                if (idx + 1) % 5 == 0:
                    logger.info(f"Completed {idx + 1}/{n_iter} iterations")
            
            except Exception as e:
                logger.warning(f"Failed with params {params}: {e}")
                continue
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {best_score:.4f}")
        
        return best_params, best_score
