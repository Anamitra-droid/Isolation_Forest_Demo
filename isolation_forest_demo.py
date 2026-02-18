"""
Isolation Forest Anomaly Detection Demo

This script demonstrates the use of Isolation Forest for anomaly detection
on a synthetic dataset with visualization and evaluation metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_data(n_samples=1000, n_features=2, contamination=0.1):
    """
    Generate synthetic dataset with normal and anomalous points.
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples
    n_features : int
        Number of features
    contamination : float
        Proportion of outliers in the dataset
    
    Returns:
    --------
    X : array-like
        Feature matrix
    y_true : array-like
        True labels (1 for normal, -1 for outliers)
    """
    # Generate normal data points
    n_normal = int(n_samples * (1 - contamination))
    X_normal, _ = make_blobs(
        n_samples=n_normal,
        centers=[[0, 0]],
        cluster_std=1.0,
        n_features=n_features,
        random_state=42
    )
    
    # Generate anomalous data points (outliers)
    n_anomalies = n_samples - n_normal
    X_anomalies = np.random.uniform(
        low=-6, high=6, size=(n_anomalies, n_features)
    )
    
    # Combine normal and anomalous data
    X = np.vstack([X_normal, X_anomalies])
    
    # Create true labels (1 for normal, -1 for outliers)
    y_true = np.ones(n_samples)
    y_true[n_normal:] = -1
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y_true = y_true[indices]
    
    return X, y_true

def train_isolation_forest(X, contamination=0.1, n_estimators=100, random_state=42):
    """
    Train Isolation Forest model.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    contamination : float
        Expected proportion of outliers
    n_estimators : int
        Number of trees in the forest
    random_state : int
        Random seed
    
    Returns:
    --------
    model : IsolationForest
        Trained Isolation Forest model
    """
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X)
    return model

def evaluate_model(model, X, y_true):
    """
    Evaluate the Isolation Forest model.
    
    Parameters:
    -----------
    model : IsolationForest
        Trained model
    X : array-like
        Feature matrix
    y_true : array-like
        True labels
    
    Returns:
    --------
    y_pred : array-like
        Predicted labels
    scores : array-like
        Anomaly scores
    """
    # Predict outliers (-1 for outliers, 1 for inliers)
    y_pred = model.predict(X)
    
    # Get anomaly scores (lower scores indicate more outlying)
    scores = model.decision_function(X)
    
    return y_pred, scores

def plot_results(X, y_true, y_pred, scores, save_path='results.png'):
    """
    Visualize the results of anomaly detection.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    scores : array-like
        Anomaly scores
    save_path : str
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: True labels
    ax1 = axes[0]
    normal_mask = y_true == 1
    anomaly_mask = y_true == -1
    ax1.scatter(X[normal_mask, 0], X[normal_mask, 1], 
                c='blue', label='Normal', alpha=0.6, s=50)
    ax1.scatter(X[anomaly_mask, 0], X[anomaly_mask, 1], 
                c='red', label='Anomaly (True)', alpha=0.6, s=50, marker='x')
    ax1.set_title('True Labels', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Predicted labels
    ax2 = axes[1]
    normal_pred_mask = y_pred == 1
    anomaly_pred_mask = y_pred == -1
    ax2.scatter(X[normal_pred_mask, 0], X[normal_pred_mask, 1], 
                c='green', label='Normal (Predicted)', alpha=0.6, s=50)
    ax2.scatter(X[anomaly_pred_mask, 0], X[anomaly_pred_mask, 1], 
                c='orange', label='Anomaly (Predicted)', alpha=0.6, s=50, marker='x')
    ax2.set_title('Predicted Labels', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Anomaly scores
    ax3 = axes[2]
    scatter = ax3.scatter(X[:, 0], X[:, 1], c=scores, 
                         cmap='RdYlGn', alpha=0.6, s=50)
    ax3.set_title('Anomaly Scores', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Feature 1')
    ax3.set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=ax3, label='Anomaly Score')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

def print_evaluation_metrics(y_true, y_pred):
    """
    Print evaluation metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    # Convert labels for classification report (1 -> 0 for normal, -1 -> 1 for anomaly)
    y_true_binary = (y_true == -1).astype(int)
    y_pred_binary = (y_pred == -1).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_true_binary, y_pred_binary, 
                                target_names=['Normal', 'Anomaly']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    print(cm)
    
    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Calculate precision, recall, F1 for anomalies
    tp = np.sum((y_true == -1) & (y_pred == -1))
    fp = np.sum((y_true == 1) & (y_pred == -1))
    fn = np.sum((y_true == -1) & (y_pred == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nAnomaly Detection Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print("="*60 + "\n")

def main():
    """
    Main function to run the Isolation Forest demo.
    """
    print("="*60)
    print("ISOLATION FOREST ANOMALY DETECTION DEMO")
    print("="*60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic dataset...")
    X, y_true = generate_data(n_samples=1000, n_features=2, contamination=0.1)
    print(f"   Dataset shape: {X.shape}")
    print(f"   Number of normal points: {np.sum(y_true == 1)}")
    print(f"   Number of anomalies: {np.sum(y_true == -1)}")
    
    # Train Isolation Forest
    print("\n2. Training Isolation Forest model...")
    model = train_isolation_forest(X, contamination=0.1, n_estimators=100)
    print("   Model trained successfully!")
    
    # Evaluate model
    print("\n3. Evaluating model...")
    y_pred, scores = evaluate_model(model, X, y_true)
    
    # Print evaluation metrics
    print_evaluation_metrics(y_true, y_pred)
    
    # Visualize results
    print("\n4. Generating visualizations...")
    plot_results(X, y_true, y_pred, scores, save_path='isolation_forest_results.png')
    
    # Print summary statistics
    print("\n5. Summary Statistics:")
    print(f"   Mean anomaly score: {np.mean(scores):.4f}")
    print(f"   Std anomaly score: {np.std(scores):.4f}")
    print(f"   Min anomaly score: {np.min(scores):.4f}")
    print(f"   Max anomaly score: {np.max(scores):.4f}")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()
