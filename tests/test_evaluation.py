"""Tests for evaluation metrics."""

import unittest
import numpy as np
from src.evaluation import AnomalyMetrics


class TestAnomalyMetrics(unittest.TestCase):
    """Test AnomalyMetrics class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.y_true = np.concatenate([np.ones(90), -np.ones(10)])
        self.y_pred = np.concatenate([np.ones(85), -np.ones(15)])
        self.scores = np.random.randn(100)
    
    def test_calculate_metrics(self):
        """Test metric calculation."""
        metrics = AnomalyMetrics.calculate_metrics(
            self.y_true, self.y_pred, self.scores
        )
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('auc_roc', metrics)
        self.assertIn('auc_pr', metrics)
        
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
    
    def test_get_roc_curve(self):
        """Test ROC curve calculation."""
        fpr, tpr, thresholds = AnomalyMetrics.get_roc_curve(
            self.y_true, self.scores
        )
        
        self.assertEqual(len(fpr), len(tpr))
        self.assertEqual(len(fpr), len(thresholds))
    
    def test_get_pr_curve(self):
        """Test PR curve calculation."""
        precision, recall, thresholds = AnomalyMetrics.get_pr_curve(
            self.y_true, self.scores
        )
        
        self.assertEqual(len(precision), len(recall))
        self.assertEqual(len(precision), len(thresholds))


if __name__ == '__main__':
    unittest.main()
