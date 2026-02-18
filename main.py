"""
Main Entry Point

Professional-grade anomaly detection pipeline with multiple algorithms,
comprehensive evaluation, and visualization.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_data
from src.models import create_detector
from src.evaluation import AnomalyMetrics
from src.visualization import AnomalyPlotter
from src.utils.config import ExperimentConfig, get_default_config
from src.utils.hyperparameter_tuning import HyperparameterTuner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('anomaly_detection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def run_single_experiment(
    config: ExperimentConfig,
    compare_models: bool = False
) -> Dict:
    """
    Run a single anomaly detection experiment.
    
    Args:
        config: Experiment configuration
        compare_models: Whether to compare multiple models
        
    Returns:
        Dictionary with results
    """
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("ANOMALY DETECTION EXPERIMENT")
    logger.info("="*70)
    
    # Load data
    logger.info("\n1. Loading data...")
    X, y_true = load_data(
        dataset_type=config.data.dataset_type,
        n_samples=config.data.n_samples,
        n_features=config.data.n_features,
        contamination=config.data.contamination,
        normalize=config.data.normalize,
        scaler_type=config.data.scaler_type,
        random_state=config.data.random_state
    )
    logger.info(f"   Dataset shape: {X.shape}")
    logger.info(f"   Anomalies: {np.sum(y_true == -1)} ({np.sum(y_true == -1)/len(y_true)*100:.1f}%)")
    
    results = {}
    
    if compare_models:
        # Compare multiple models
        detector_types = [
            'isolation_forest',
            'elliptic_envelope',
            'lof',
            'one_class_svm'
        ]
        
        all_metrics = {}
        plotter = AnomalyPlotter()
        
        for detector_type in detector_types:
            logger.info(f"\n2. Training {detector_type}...")
            
            try:
                # Create detector
                detector = create_detector(
                    detector_type,
                    contamination=config.model.contamination,
                    random_state=config.model.random_state,
                    n_jobs=config.model.n_jobs
                )
                
                # Train
                detector.fit(X)
                
                # Predict
                y_pred = detector.predict(X)
                scores = detector.score_samples(X)
                
                # Evaluate
                metrics = AnomalyMetrics.calculate_metrics(y_true, y_pred, scores)
                all_metrics[detector.name] = metrics
                
                AnomalyMetrics.print_metrics(metrics, detector.name)
                
                # Visualize
                if config.visualization.get('plot_results', True):
                    plotter.plot_results(
                        X, y_true, y_pred, scores,
                        model_name=detector.name,
                        save_path=str(output_dir / f'{detector_type}_results.png') if config.visualization.get('save_plots') else None
                    )
                
                results[detector_type] = {
                    'metrics': metrics,
                    'predictions': y_pred,
                    'scores': scores
                }
                
            except Exception as e:
                logger.error(f"Error with {detector_type}: {e}")
                continue
        
        # Plot comparison
        if config.visualization.get('plot_comparison', True) and all_metrics:
            plotter.plot_model_comparison(
                all_metrics,
                save_path=str(output_dir / 'model_comparison.png') if config.visualization.get('save_plots') else None
            )
    
    else:
        # Single model experiment
        logger.info(f"\n2. Training {config.model.detector_type}...")
        
        # Create detector
        detector = create_detector(
            config.model.detector_type,
            contamination=config.model.contamination,
            random_state=config.model.random_state,
            n_jobs=config.model.n_jobs,
            n_estimators=config.model.n_estimators
        )
        
        # Train
        detector.fit(X)
        
        # Predict
        y_pred = detector.predict(X)
        scores = detector.score_samples(X)
        
        # Evaluate
        metrics = AnomalyMetrics.calculate_metrics(y_true, y_pred, scores)
        AnomalyMetrics.print_metrics(metrics, detector.name)
        
        # Visualize
        plotter = AnomalyPlotter()
        
        if config.visualization.get('plot_results', True):
            plotter.plot_results(
                X, y_true, y_pred, scores,
                model_name=detector.name,
                save_path=str(output_dir / 'results.png') if config.visualization.get('save_plots') else None
            )
        
        if config.evaluation.get('calculate_roc', True):
            fpr, tpr, _ = AnomalyMetrics.get_roc_curve(y_true, scores)
            plotter.plot_roc_curve(
                fpr, tpr, metrics['auc_roc'],
                model_name=detector.name,
                save_path=str(output_dir / 'roc_curve.png') if config.visualization.get('save_plots') else None
            )
        
        if config.evaluation.get('calculate_pr', True):
            precision, recall, _ = AnomalyMetrics.get_pr_curve(y_true, scores)
            plotter.plot_pr_curve(
                precision, recall, metrics['auc_pr'],
                model_name=detector.name,
                save_path=str(output_dir / 'pr_curve.png') if config.visualization.get('save_plots') else None
            )
        
        if config.visualization.get('plot_score_distribution', True):
            plotter.plot_score_distribution(
                scores, y_true,
                model_name=detector.name,
                save_path=str(output_dir / 'score_distribution.png') if config.visualization.get('save_plots') else None
            )
        
        results = {
            'metrics': metrics,
            'predictions': y_pred,
            'scores': scores,
            'model': detector
        }
    
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
    logger.info("="*70)
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Professional Anomaly Detection Pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (JSON or YAML)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['blob', 'moon', 'circle', 'high_dim'],
        default='blob',
        help='Dataset type'
    )
    parser.add_argument(
        '--detector',
        type=str,
        choices=['isolation_forest', 'elliptic_envelope', 'lof', 'one_class_svm', 'dbscan'],
        default='isolation_forest',
        help='Anomaly detector type'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple models'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=1000,
        help='Number of samples'
    )
    parser.add_argument(
        '--contamination',
        type=float,
        default=0.1,
        help='Contamination rate'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = ExperimentConfig.load(args.config)
    else:
        config = get_default_config()
        config.data.dataset_type = args.dataset
        config.model.detector_type = args.detector
        config.data.n_samples = args.samples
        config.data.contamination = args.contamination
        config.model.contamination = args.contamination
        config.output_dir = args.output_dir
    
    # Run experiment
    results = run_single_experiment(config, compare_models=args.compare)
    
    # Save config
    config_path = Path(config.output_dir) / 'config.json'
    config.save(str(config_path))


if __name__ == "__main__":
    main()
