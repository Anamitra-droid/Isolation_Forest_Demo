"""
Visualization Module

Create comprehensive visualizations for anomaly detection results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


class AnomalyPlotter:
    """Create visualizations for anomaly detection."""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 5)):
        """
        Initialize plotter.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
    
    def plot_results(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        scores: np.ndarray,
        model_name: str = "",
        save_path: Optional[str] = None
    ):
        """
        Plot comprehensive results visualization.
        
        Args:
            X: Feature matrix
            y_true: True labels
            y_pred: Predicted labels
            scores: Anomaly scores
            model_name: Name of the model
            save_path: Path to save the plot
        """
        if X.shape[1] > 2:
            logger.warning("Data has more than 2 features. Using PCA for visualization.")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_plot = pca.fit_transform(X)
        else:
            X_plot = X
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: True labels
        self._plot_labels(
            axes[0], X_plot, y_true,
            title='True Labels',
            colors=['blue', 'red']
        )
        
        # Plot 2: Predicted labels
        self._plot_labels(
            axes[1], X_plot, y_pred,
            title='Predicted Labels',
            colors=['green', 'orange']
        )
        
        # Plot 3: Anomaly scores
        scatter = axes[2].scatter(
            X_plot[:, 0], X_plot[:, 1],
            c=scores, cmap='RdYlGn_r',
            alpha=0.6, s=50, edgecolors='black', linewidth=0.5
        )
        axes[2].set_title('Anomaly Scores', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Feature 1' if X.shape[1] <= 2 else 'PC1')
        axes[2].set_ylabel('Feature 2' if X.shape[1] <= 2 else 'PC2')
        plt.colorbar(scatter, ax=axes[2], label='Anomaly Score')
        axes[2].grid(True, alpha=0.3)
        
        if model_name:
            fig.suptitle(f'{model_name} - Anomaly Detection Results',
                        fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def _plot_labels(
        self,
        ax,
        X: np.ndarray,
        y: np.ndarray,
        title: str,
        colors: List[str]
    ):
        """Helper to plot labels."""
        normal_mask = y == 1
        anomaly_mask = y == -1
        
        ax.scatter(
            X[normal_mask, 0], X[normal_mask, 1],
            c=colors[0], label='Normal', alpha=0.6, s=50,
            edgecolors='black', linewidth=0.5
        )
        ax.scatter(
            X[anomaly_mask, 0], X[anomaly_mask, 1],
            c=colors[1], label='Anomaly', alpha=0.6, s=50,
            marker='x', linewidths=2
        )
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Feature 1' if X.shape[1] <= 2 else 'PC1')
        ax.set_ylabel('Feature 2' if X.shape[1] <= 2 else 'PC2')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc: float,
        model_name: str = "",
        save_path: Optional[str] = None
    ):
        """Plot ROC curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}' if model_name else 'ROC Curve',
                 fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_pr_curve(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        auc: float,
        model_name: str = "",
        save_path: Optional[str] = None
    ):
        """Plot Precision-Recall curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'PR (AUC = {auc:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {model_name}' if model_name else 'Precision-Recall Curve',
                 fontsize=14, fontweight='bold')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR curve saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of multiple models.
        
        Args:
            metrics_dict: Dictionary mapping model names to their metrics
            save_path: Path to save the plot
        """
        models = list(metrics_dict.keys())
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        
        fig, axes = plt.subplots(1, len(metric_names), figsize=(20, 4))
        
        for idx, metric in enumerate(metric_names):
            values = [metrics_dict[model][metric] for model in models]
            axes[idx].bar(models, values, alpha=0.7, edgecolor='black')
            axes[idx].set_title(metric.replace('_', ' ').title(), fontweight='bold')
            axes[idx].set_ylabel('Score')
            axes[idx].set_ylim([0, 1])
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Model Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_score_distribution(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        model_name: str = "",
        save_path: Optional[str] = None
    ):
        """Plot distribution of anomaly scores."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        normal_scores = scores[y_true == 1]
        anomaly_scores = scores[y_true == -1]
        
        # Histogram
        axes[0].hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue')
        axes[0].hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red')
        axes[0].set_xlabel('Anomaly Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Score Distribution', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot([normal_scores, anomaly_scores], labels=['Normal', 'Anomaly'])
        axes[1].set_ylabel('Anomaly Score')
        axes[1].set_title('Score Distribution (Box Plot)', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        if model_name:
            fig.suptitle(f'{model_name} - Score Distribution',
                        fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Score distribution plot saved to {save_path}")
        
        plt.show()
