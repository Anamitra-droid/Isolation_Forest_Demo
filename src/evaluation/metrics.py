"""
Evaluation Metrics Module

Comprehensive evaluation metrics for anomaly detection.
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
import logging

logger = logging.getLogger(__name__)


class AnomalyMetrics:
    """Calculate comprehensive metrics for anomaly detection."""
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        scores: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels (1 for normal, -1 for anomaly)
            y_pred: Predicted labels (1 for normal, -1 for anomaly)
            scores: Anomaly scores
            
        Returns:
            Dictionary of metrics
        """
        # Convert to binary (0 for normal, 1 for anomaly)
        y_true_binary = (y_true == -1).astype(int)
        y_pred_binary = (y_pred == -1).astype(int)
        
        # Basic metrics
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # ROC and PR metrics
        try:
            auc_roc = roc_auc_score(y_true_binary, scores)
        except ValueError:
            auc_roc = 0.0
        
        try:
            auc_pr = average_precision_score(y_true_binary, scores)
        except ValueError:
            auc_pr = 0.0
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = recall  # Same as recall
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
        
        return metrics
    
    @staticmethod
    def get_roc_curve(
        y_true: np.ndarray,
        scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get ROC curve data.
        
        Args:
            y_true: True labels
            scores: Anomaly scores
            
        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        y_true_binary = (y_true == -1).astype(int)
        fpr, tpr, thresholds = roc_curve(y_true_binary, scores)
        return fpr, tpr, thresholds
    
    @staticmethod
    def get_pr_curve(
        y_true: np.ndarray,
        scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get Precision-Recall curve data.
        
        Args:
            y_true: True labels
            scores: Anomaly scores
            
        Returns:
            Tuple of (precision, recall, thresholds)
        """
        y_true_binary = (y_true == -1).astype(int)
        precision, recall, thresholds = precision_recall_curve(
            y_true_binary, scores
        )
        return precision, recall, thresholds
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], model_name: str = ""):
        """
        Print metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metrics
            model_name: Name of the model
        """
        print("\n" + "="*70)
        if model_name:
            print(f"METRICS FOR {model_name.upper()}")
        else:
            print("EVALUATION METRICS")
        print("="*70)
        
        print(f"\nClassification Metrics:")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1-Score:    {metrics['f1_score']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        
        print(f"\nArea Under Curves:")
        print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR:      {metrics['auc_pr']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        
        print("="*70 + "\n")
    
    @staticmethod
    def get_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Get sklearn classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report string
        """
        y_true_binary = (y_true == -1).astype(int)
        y_pred_binary = (y_pred == -1).astype(int)
        return classification_report(
            y_true_binary,
            y_pred_binary,
            target_names=['Normal', 'Anomaly']
        )
