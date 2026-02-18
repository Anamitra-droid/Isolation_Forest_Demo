"""Tests for anomaly detection models."""

import unittest
import numpy as np
from src.models import create_detector
from src.data import load_data


class TestAnomalyDetectors(unittest.TestCase):
    """Test anomaly detector models."""
    
    def setUp(self):
        """Set up test data."""
        self.X, self.y = load_data('blob', n_samples=200, normalize=True)
    
    def test_isolation_forest(self):
        """Test Isolation Forest detector."""
        detector = create_detector('isolation_forest', contamination=0.1)
        detector.fit(self.X)
        predictions = detector.predict(self.X)
        scores = detector.score_samples(self.X)
        
        self.assertEqual(len(predictions), len(self.X))
        self.assertEqual(len(scores), len(self.X))
        self.assertTrue(np.all(np.isin(predictions, [-1, 1])))
    
    def test_elliptic_envelope(self):
        """Test Elliptic Envelope detector."""
        detector = create_detector('elliptic_envelope', contamination=0.1)
        detector.fit(self.X)
        predictions = detector.predict(self.X)
        scores = detector.score_samples(self.X)
        
        self.assertEqual(len(predictions), len(self.X))
        self.assertEqual(len(scores), len(self.X))
    
    def test_lof(self):
        """Test Local Outlier Factor detector."""
        detector = create_detector('lof', contamination=0.1, novelty=True)
        detector.fit(self.X)
        predictions = detector.predict(self.X)
        scores = detector.score_samples(self.X)
        
        self.assertEqual(len(predictions), len(self.X))
        self.assertEqual(len(scores), len(self.X))
    
    def test_one_class_svm(self):
        """Test One-Class SVM detector."""
        detector = create_detector('one_class_svm', nu=0.1)
        detector.fit(self.X)
        predictions = detector.predict(self.X)
        scores = detector.score_samples(self.X)
        
        self.assertEqual(len(predictions), len(self.X))
        self.assertEqual(len(scores), len(self.X))
    
    def test_invalid_detector(self):
        """Test invalid detector type."""
        with self.assertRaises(ValueError):
            create_detector('invalid_detector')


if __name__ == '__main__':
    unittest.main()
