"""
Re-Evaluate Hybrid Model at Custom Decision Thresholds & Lambda Boosts
========================================================================
Uses the SAVED test predictions (log_odds, heuristic fields) from previous inference.
No model loading or GPU required — pure NumPy re-computation of Samsung Decision Engine.

Usage:
  python 7_re_evaluate_thresholds.py
  python 7_re_evaluate_thresholds.py --thresholds 0.5 0.55 0.6 --lambdas 0.2 0.5 0.8
  python 7_re_evaluate_thresholds.py --predictions path/to/test_predictions.csv
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Ensure sibling packages under SRC can be imported regardless of run directory
sys.path.append(str(Path(__file__).resolve().parent))

from core.config import Config
from core.evaluator import EnhancedKPIEvaluator, SamsungDecisionEngine


def print_results(metrics: dict, label: str = ""):
    """Pretty-print results for a single threshold configuration."""
    t = metrics['threshold']
    lam = metrics['lambda']
    
    print(f"\n{'='*70}")
    print(f"  BASE THRESHOLD = {t:.3f} | LAMBDA BOOST = {lam:.2f}  {label}")
    print(f"{'='*70}")
    
    print(f"\n  {'Metric':<15} {'Value':>10} {'Target':>12} {'Status':>8}")
    print(f"  {'-'*47}")
    
    rows = [
        ('Accuracy',  metrics['accuracy'],  f'>= {Config.TARGET_ACCURACY:.0%}',  metrics['kpi_checks']['accuracy_met']),
        ('Precision', metrics['precision'], f'>= {Config.TARGET_PRECISION:.0%}', metrics['kpi_checks']['precision_met']),
        ('Recall',    metrics['recall'],    f'>= {Config.TARGET_RECALL:.0%}',    metrics['kpi_checks']['recall_met']),
        ('F1-Score',  metrics['f1'],        '—',                          None),
        ('AUC-ROC',   metrics['auc'],       '—',                          None),
        ('FNR',       metrics['fnr'],       f'<= {Config.MAX_FNR:.0%}',         metrics['kpi_checks']['fnr_met']),
        ('FPR',       metrics['fpr'],       f'<= {Config.MAX_FPR:.0%}',         metrics['kpi_checks']['fpr_met']),
    ]
    
    for name, value, target, passed in rows:
        if passed is None:
            status = "  "
        elif passed:
            status = "[OK]"
        else:
            status = "[FAIL]"
        print(f"  {name:<15} {value:>10.4f} {target:>12} {status:>8}")
    
    print(f"\n  Confusion Matrix:")
    print(f"  +-------------------------------------------+")
    print(f"  |              Predicted Benign   Predicted  |")
    print(f"  |                (Label 0)      Malicious    |")
    print(f"  |                               (Label 1)   |")
    print(f"  +-------------------------------------------+")
    print(f"  |  Actual Benign    TN={metrics['tn']:>10,}  FP={metrics['fp']:>9,}  |")
    print(f"  |  Actual Malicious FN={metrics['fn']:>10,}  TP={metrics['tp']:>9,}  |")
    print(f"  +-------------------------------------------+")
    
    status_str = "[PASS] ALL KPIs MET" if metrics['kpi_compliance'] else "[FAIL] KPIs NOT MET"
    print(f"\n  Overall KPI Compliance: {status_str}")
    print(f"{'='*70}")


def print_comparison_table(all_metrics: list):
    """Print a comparison table across all evaluated points."""
    print(f"\n\n{'#'*95}")
    print(f"{'  HYBRID OPERATING POINT COMPARISON SUMMARY':^95}")
    print(f"{'#'*95}\n")
    
    header = f"  {'Metric':<12}"
    for m in all_metrics:
        pt_label = f"t={m['threshold']:.3f}/L={m['lambda']:.1f}"
        header += f" {pt_label:>15}"
    print(header)
    print(f"  {'-' * (12 + 16 * len(all_metrics))}")
    
    metrics_to_show = [
        ('Accuracy',  'accuracy'),
        ('Precision', 'precision'),
        ('Recall',    'recall'),
        ('F1-Score',  'f1'),
        ('AUC-ROC',   'auc'),
        ('FPR',       'fpr'),
        ('FNR',       'fnr'),
        ('TN',        'tn'),
        ('FP',        'fp'),
        ('FN',        'fn'),
        ('TP',        'tp'),
    ]
    
    for name, key in metrics_to_show:
        row = f"  {name:<12}"
        for m in all_metrics:
            val = m[key]
            if isinstance(val, int):
                row += f" {val:>15,}"
            else:
                row += f" {val:>15.4f}"
        print(row)
    
    # KPI compliance row
    row = f"  {'KPI Pass':<12}"
    for m in all_metrics:
        status = "YES" if m['kpi_compliance'] else "NO"
        row += f" {status:>15}"
    print(row)
    
    print(f"\n  KPI Targets: Accuracy >= {Config.TARGET_ACCURACY:.0%}, Precision >= {Config.TARGET_PRECISION:.0%}, "
          f"Recall >= {Config.TARGET_RECALL:.0%}, FPR <= {Config.MAX_FPR:.1%}, FNR <= {Config.MAX_FNR:.1%}")
    print(f"{'#'*95}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate Hybrid model offline across custom threshold-lambda dimensions.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config', type=str, default='4_config.yaml',
        help="Path to centralized configuration file (default: 4_config.yaml)"
    )
    
    parser.add_argument(
        '--predictions', type=str, 
        default=None,
        help="Path to test_predictions.csv from inference output"
    )
    
    parser.add_argument(
        '--thresholds', type=float, nargs='+',
        default=[0.45, 0.50, 0.55],
        help="List of base thresholds to evaluate (default: 0.45 0.50 0.55)"
    )
    
    parser.add_argument(
        '--lambdas', type=float, nargs='+',
        default=[0.0, 0.5, 1.0],
        help="List of lambda risk coefficients to evaluate (default: 0.0 0.5 1.0)"
    )
    
    parser.add_argument(
        '--output', type=str, default=None,
        help="Path to save comparison results JSON (optional)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    Config.load_from_yaml(args.config)
    
    # Resolve default predictions path from config
    pred_path = args.predictions
    if pred_path is None:
        pred_path = str(Config.SAVE_ROOT / "final_test_evaluation" / "test_predictions.csv")
    
    pred_path_obj = Path(pred_path)
    if not pred_path_obj.exists():
        print(f"❌ Predictions file not found: {pred_path_obj}")
        print(f"   Run training or inference first to dump test_predictions.csv")
        return
    
    print(f"\n{'='*70}")
    print(f"  HYBRID JOINT CALIBRATION RE-EVALUATION TOOL")
    print(f"{'='*70}")
    print(f"  Predictions file:       {pred_path_obj.name}")
    print(f"  Thresholds to evaluate: {args.thresholds}")
    print(f"  Lambdas to evaluate:    {args.lambdas}")
    
    df = pd.read_csv(pred_path_obj)
    print(f"  Total samples:          {len(df):,}")
    
    required_cols = ['true_label', 'log_odds', 'h_severity_score', 'h_flags_count', 'h_primary_category']
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ ERROR: Missing required column '{col}' in predictions CSV!")
            return
            
    # Clean up potential string representation of PyTorch Tensors (e.g., "tensor(6., dtype=torch.float64)" or "tensor(2)")
    def clean_tensor_column(col_series):
        def clean_val(v):
            if not isinstance(v, str):
                return float(v)
            if 'tensor(' in v:
                v = v.replace('tensor(', '').replace(')', '')
                if ',' in v:
                    v = v.split(',')[0]
            try:
                return float(v)
            except ValueError:
                return 0.0
        return col_series.apply(clean_val).values

    y_true = df['true_label'].values
    log_odds = df['log_odds'].values
    severities = clean_tensor_column(df['h_severity_score'])
    flags_counts = clean_tensor_column(df['h_flags_count'])
    categories = df['h_primary_category'].values
    
    # Extract calibrated probabilities for ROC/PR calculations (prob = Sigmoid(log_odds))
    y_prob = 1.0 / (1.0 + np.exp(-log_odds))
    
    label_dist = pd.Series(y_true).value_counts().to_dict()
    print(f"  Label distribution:     {label_dist}")
    print(f"  Log-odds range:         [{log_odds.min():.4f}, {log_odds.max():.4f}]")
    print(f"  Probability range:      [{y_prob.min():.4f}, {y_prob.max():.4f}]")
    print(f"{'='*70}")
    
    evaluator = EnhancedKPIEvaluator()
    all_results = []
    
    for lambda_val in sorted(args.lambdas):
        for threshold in sorted(args.thresholds):
            y_pred = SamsungDecisionEngine.decide(
                log_odds=log_odds,
                severities=severities,
                flags_counts=flags_counts,
                categories=categories,
                base_threshold=threshold,
                lambda_val=lambda_val
            )
            
            metrics = evaluator.evaluate_metrics(y_true, y_pred, y_prob)
            metrics['threshold'] = threshold
            metrics['lambda'] = lambda_val
            
            print_results(metrics)
            all_results.append(metrics)
            
    # Comparison table
    print_comparison_table(all_results)
    
    # Save results
    output_path = args.output
    if output_path is None:
        output_path = pred_path_obj.parent / "joint_threshold_comparison_results.json"
    
    output_data = {
        'predictions_file': str(pred_path_obj),
        'total_samples': len(df),
        'label_distribution': {str(k): int(v) for k, v in label_dist.items()},
        'results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"  [OK] Joint comparison results saved to: {output_path.name}\n")


if __name__ == "__main__":
    main()
