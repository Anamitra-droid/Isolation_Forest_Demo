"""Tests for data loading module."""

import unittest
import numpy as np
from src.data import DataGenerator, DataPreprocessor, load_data


class TestDataGenerator(unittest.TestCase):
    """Test DataGenerator class."""
    
    def test_generate_blob_data(self):
        """Test blob data generation."""
        X, y = DataGenerator.generate_blob_data(
            n_samples=100, n_features=2, contamination=0.1
        )
        self.assertEqual(X.shape, (100, 2))
        self.assertEqual(len(y), 100)
        self.assertEqual(np.sum(y == -1), 10)  # 10% anomalies
    
    def test_generate_moon_data(self):
        """Test moon data generation."""
        X, y = DataGenerator.generate_moon_data(n_samples=100, contamination=0.1)
        self.assertEqual(X.shape, (100, 2))
        self.assertEqual(len(y), 100)
    
    def test_generate_circle_data(self):
        """Test circle data generation."""
        X, y = DataGenerator.generate_circle_data(n_samples=100, contamination=0.1)
        self.assertEqual(X.shape, (100, 2))
        self.assertEqual(len(y), 100)
    
    def test_generate_high_dimensional_data(self):
        """Test high-dimensional data generation."""
        X, y = DataGenerator.generate_high_dimensional_data(
            n_samples=100, n_features=10, contamination=0.1
        )
        self.assertEqual(X.shape, (100, 10))
        self.assertEqual(len(y), 100)


class TestDataPreprocessor(unittest.TestCase):
    """Test DataPreprocessor class."""
    
    def test_standard_scaler(self):
        """Test standard scaler."""
        preprocessor = DataPreprocessor(scaler_type='standard')
        X = np.random.randn(100, 2) * 10 + 5
        X_scaled = preprocessor.fit_transform(X)
        self.assertAlmostEqual(np.mean(X_scaled), 0, places=1)
        self.assertAlmostEqual(np.std(X_scaled), 1, places=1)
    
    def test_robust_scaler(self):
        """Test robust scaler."""
        preprocessor = DataPreprocessor(scaler_type='robust')
        X = np.random.randn(100, 2) * 10 + 5
        X_scaled = preprocessor.fit_transform(X)
        self.assertEqual(X_scaled.shape, X.shape)


class TestLoadData(unittest.TestCase):
    """Test load_data function."""
    
    def test_load_blob_data(self):
        """Test loading blob data."""
        X, y = load_data('blob', n_samples=100, normalize=False)
        self.assertEqual(X.shape[0], 100)
        self.assertEqual(len(y), 100)
    
    def test_load_with_normalization(self):
        """Test loading with normalization."""
        X, y = load_data('blob', n_samples=100, normalize=True)
        self.assertEqual(X.shape[0], 100)


if __name__ == '__main__':
    unittest.main()
