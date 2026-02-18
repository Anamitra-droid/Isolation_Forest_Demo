# Isolation Forest Anomaly Detection

[![CI](https://github.com/Sourav692/Isolation_Forest_Demo/actions/workflows/ci.yml/badge.svg)](https://github.com/Sourav692/Isolation_Forest_Demo/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, production-ready anomaly detection system implementing Isolation Forest and multiple state-of-the-art algorithms with advanced evaluation metrics, visualization, and professional software engineering practices.

## 🎯 Features

### Core Capabilities
- **Multiple Anomaly Detection Algorithms**:
  - Isolation Forest
  - Elliptic Envelope
  - Local Outlier Factor (LOF)
  - One-Class SVM
  - DBSCAN

- **Comprehensive Data Generation**:
  - Blob-shaped datasets
  - Moon-shaped datasets
  - Circle-shaped datasets
  - High-dimensional datasets
  - Customizable contamination rates

- **Advanced Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-Score
  - Specificity, Sensitivity
  - AUC-ROC and AUC-PR curves
  - Confusion matrix analysis
  - Classification reports

- **Professional Visualizations**:
  - Multi-panel result comparisons
  - ROC and Precision-Recall curves
  - Score distribution analysis
  - Model comparison charts
  - PCA-based visualization for high-dimensional data

- **Production-Ready Features**:
  - Modular architecture with separation of concerns
  - Configuration management (YAML/JSON)
  - Comprehensive logging
  - Unit tests with coverage
  - Docker support
  - CI/CD pipeline (GitHub Actions)
  - CLI interface
  - Hyperparameter tuning utilities

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Examples](#examples)
- [Testing](#testing)
- [Docker](#docker)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/Sourav692/Isolation_Forest_Demo.git
cd Isolation_Forest_Demo

# Install dependencies
pip install -r requirements.txt
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Or install manually
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 isort mypy
```

## ⚡ Quick Start

### Basic Usage

Run a single experiment with default settings:

```bash
python main.py
```

### Compare Multiple Models

```bash
python main.py --compare
```

### Use Configuration File

```bash
python main.py --config configs/default_config.yaml
```

### Command-Line Options

```bash
python main.py --help
```

Available options:
- `--config`: Path to configuration file (YAML or JSON)
- `--dataset`: Dataset type (blob, moon, circle, high_dim)
- `--detector`: Detector type (isolation_forest, elliptic_envelope, lof, one_class_svm, dbscan)
- `--compare`: Compare multiple models
- `--samples`: Number of samples
- `--contamination`: Contamination rate (0.0-1.0)
- `--output-dir`: Output directory for results

## 📖 Usage

### Python API

```python
from src.data import load_data
from src.models import create_detector
from src.evaluation import AnomalyMetrics
from src.visualization import AnomalyPlotter

# Load data
X, y_true = load_data('blob', n_samples=1000, contamination=0.1)

# Create and train detector
detector = create_detector('isolation_forest', contamination=0.1)
detector.fit(X)

# Predict
y_pred = detector.predict(X)
scores = detector.score_samples(X)

# Evaluate
metrics = AnomalyMetrics.calculate_metrics(y_true, y_pred, scores)
AnomalyMetrics.print_metrics(metrics, 'Isolation Forest')

# Visualize
plotter = AnomalyPlotter()
plotter.plot_results(X, y_true, y_pred, scores, model_name='Isolation Forest')
```

### Configuration Files

Create custom configuration files in YAML or JSON format:

```yaml
data:
  dataset_type: blob
  n_samples: 1000
  n_features: 2
  contamination: 0.1
  normalize: true
  scaler_type: standard

model:
  detector_type: isolation_forest
  contamination: 0.1
  n_estimators: 100
  random_state: 42

evaluation:
  calculate_roc: true
  calculate_pr: true

visualization:
  plot_results: true
  plot_roc: true
  plot_pr: true
  save_plots: true

output_dir: outputs
```

## 📁 Project Structure

```
Isolation_Forest_Demo/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py          # Data generation and preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   └── anomaly_detectors.py   # Anomaly detection algorithms
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py              # Evaluation metrics
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plotter.py              # Visualization utilities
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # Configuration management
│       └── hyperparameter_tuning.py # Hyperparameter optimization
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_models.py
│   └── test_evaluation.py
├── configs/
│   ├── default_config.yaml
│   └── comparison_config.yaml
├── outputs/                        # Generated outputs (gitignored)
├── main.py                         # Main entry point
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
├── Makefile                        # Convenience commands
├── Dockerfile                      # Docker configuration
├── .github/
│   └── workflows/
│       └── ci.yml                  # CI/CD pipeline
└── README.md                       # This file
```

## ⚙️ Configuration

### Data Configuration

- `dataset_type`: Type of synthetic dataset (blob, moon, circle, high_dim)
- `n_samples`: Total number of samples
- `n_features`: Number of features (for blob and high_dim)
- `contamination`: Proportion of anomalies (0.0-1.0)
- `normalize`: Whether to normalize data
- `scaler_type`: Scaler type (standard, robust)

### Model Configuration

#### Isolation Forest
- `contamination`: Expected proportion of outliers
- `n_estimators`: Number of trees in the forest
- `max_samples`: Number of samples for each tree
- `max_features`: Number of features for each tree
- `bootstrap`: Whether to use bootstrap sampling

#### Other Models
See individual model documentation in `src/models/anomaly_detectors.py`

## 📊 Examples

### Example 1: Single Model Experiment

```bash
python main.py --dataset blob --detector isolation_forest --samples 1000 --contamination 0.1
```

### Example 2: Model Comparison

```bash
python main.py --compare --dataset moon --samples 2000
```

### Example 3: High-Dimensional Data

```bash
python main.py --dataset high_dim --n_features 20 --detector isolation_forest
```

### Example 4: Custom Configuration

```bash
python main.py --config configs/comparison_config.yaml
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

### Test Coverage

The project includes comprehensive unit tests covering:
- Data generation and preprocessing
- All anomaly detection models
- Evaluation metrics
- Edge cases and error handling

## 🐳 Docker

### Build Docker Image

```bash
docker build -t isolation-forest-demo .
```

### Run Container

```bash
docker run --rm -v $(pwd)/outputs:/app/outputs isolation-forest-demo
```

### Using Makefile

```bash
make docker-build
make docker-run
```

## 🛠️ Development

### Code Quality

```bash
# Format code
make format
# or
black src/ tests/ main.py
isort src/ tests/ main.py

# Lint code
make lint
# or
flake8 src/ tests/ main.py
```

### Makefile Commands

```bash
make install      # Install dependencies
make test         # Run tests
make lint         # Run linting
make format       # Format code
make run          # Run main script
make clean        # Clean generated files
make docker-build # Build Docker image
make docker-run   # Run Docker container
```

## 📈 How Isolation Forest Works

Isolation Forest is an unsupervised anomaly detection algorithm that works on the principle of isolating anomalies rather than profiling normal data points.

### Algorithm Overview

1. **Random Partitioning**: The algorithm randomly selects a feature and a split value between the max and min of that feature.

2. **Tree Construction**: This process creates a tree structure where:
   - Normal points require more partitions to isolate
   - Anomalies require fewer partitions (shorter paths)

3. **Forest Ensemble**: Multiple trees are created, and the average path length across all trees determines the anomaly score.

4. **Anomaly Detection**: Points with shorter average path lengths are more likely to be anomalies.

### Advantages

- Efficient for high-dimensional data
- Handles large datasets well
- No need for labeled data
- Works well with mixed data types
- Fast training and prediction

## 📚 References

- [scikit-learn Isolation Forest Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [Isolation Forest Paper (Liu et al., 2008)](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
- [Anomaly Detection Survey](https://www.cs.umn.edu/~kumar001/dmbook/ch10.pdf)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Write tests for new features
- Update documentation
- Ensure all tests pass
- Run linting before committing

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Sourav Banerjee**

- GitHub: [@Sourav692](https://github.com/Sourav692)
- LinkedIn: [Sourav Banerjee](https://www.linkedin.com/in/sourav-banerjee-50b443106/)

## 🙏 Acknowledgments

- scikit-learn team for the excellent library
- Contributors to the Isolation Forest algorithm
- Open source community

## 📧 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

⭐ If you find this project useful, please consider giving it a star!
