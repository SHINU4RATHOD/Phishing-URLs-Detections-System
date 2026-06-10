"""
Re-Evaluate Model at Custom Decision Thresholds
================================================
Uses the SAVED test predictions (prob_malicious) from previous inference.
No model loading or GPU required — pure NumPy re-computation.

Usage:
  python 7_re_evaluate_thresholds.py
  python 7_re_evaluate_thresholds.py --thresholds 0.5 0.525 0.7 0.9 0.999
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
from core.evaluator import EnhancedKPIEvaluator


def print_results(metrics: dict, label: str = ""):
    """Pretty-print results for a single threshold."""
    t = metrics['threshold']
    
    print(f"\n{'='*70}")
    print(f"  THRESHOLD = {t:.4f}  {label}")
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
    """Print a side-by-side comparison table across all thresholds."""
    print(f"\n\n{'#'*80}")
    print(f"{'  THRESHOLD COMPARISON SUMMARY':^80}")
    print(f"{'#'*80}\n")
    
    header = f"  {'Metric':<12}"
    for m in all_metrics:
        t_label = "t=" + f"{m['threshold']:.3f}"
        header += f" {t_label:>12}"
    print(header)
    print(f"  {'-' * (12 + 13 * len(all_metrics))}")
    
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
                row += f" {val:>12,}"
            else:
                row += f" {val:>12.4f}"
        print(row)
    
    # KPI compliance row
    row = f"  {'KPI Pass':<12}"
    for m in all_metrics:
        status = "YES" if m['kpi_compliance'] else "NO"
        row += f" {status:>12}"
    print(row)
    
    print(f"\n  KPI Targets: Accuracy >= {Config.TARGET_ACCURACY:.0%}, Precision >= {Config.TARGET_PRECISION:.0%}, "
          f"Recall >= {Config.TARGET_RECALL:.0%}, FPR <= {Config.MAX_FPR:.0%}, FNR <= {Config.MAX_FNR:.0%}")
    print(f"{'#'*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate model at custom decision thresholds (no GPU needed)",
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
        default=[0.525, 0.90, 0.999],
        help="List of thresholds to evaluate (default: 0.525 0.90 0.999)"
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
        print(f"   Run inference first: python 6_inference.py --mode inference")
        return
    
    print(f"\n{'='*70}")
    print(f"  THRESHOLD RE-EVALUATION TOOL")
    print(f"{'='*70}")
    print(f"  Predictions file: {pred_path_obj.name}")
    print(f"  Thresholds to evaluate: {args.thresholds}")
    
    df = pd.read_csv(pred_path_obj)
    print(f"  Total samples: {len(df):,}")
    
    y_true = df['true_label'].values
    y_prob = df['prob_malicious'].values
    
    label_dist = pd.Series(y_true).value_counts().to_dict()
    print(f"  Label distribution: {label_dist}")
    print(f"  Probability range: [{y_prob.min():.6f}, {y_prob.max():.6f}]")
    print(f"  Probability mean:  {y_prob.mean():.6f}")
    print(f"  Probability median: {np.median(y_prob):.6f}")
    print(f"{'='*70}")
    
    # Evaluate at each threshold
    evaluator = EnhancedKPIEvaluator()
    all_results = []
    
    for threshold in sorted(args.thresholds):
        y_pred = (y_prob >= threshold).astype(int)
        metrics = evaluator.evaluate_metrics(y_true, y_pred, y_prob)
        metrics['threshold'] = threshold
        
        label = "(ORIGINAL)" if abs(threshold - 0.525) < 0.001 else ""
        print_results(metrics, label)
        all_results.append(metrics)
    
    # Comparison table
    print_comparison_table(all_results)
    
    # Analysis & interpretation
    print(f"{'='*70}")
    print(f"  ANALYSIS & INTERPRETATION")
    print(f"{'='*70}\n")
    
    for m in all_results:
        t = m['threshold']
        print(f"  Threshold {t:.3f}:")
        
        if t < 0.6:
            print(f"    -> Standard threshold. Balanced between FPR and FNR.")
            print(f"    -> FPR={m['fpr']:.2%} means {m['fp']:,} benign URLs are flagged as phishing.")
            print(f"    -> FNR={m['fnr']:.2%} means {m['fn']:,} phishing URLs slip through.")
        elif t >= 0.9 and t < 0.99:
            print(f"    -> HIGH-CONFIDENCE threshold. Only flags URLs the model is >= {t:.0%} sure are phishing.")
            print(f"    -> FPR drops to {m['fpr']:.2%} - fewer false alarms for security teams.")
            print(f"    -> FNR rises to {m['fnr']:.2%} - more phishing URLs slip through ({m['fn']:,} missed).")
            print(f"    -> Trade-off: Precision increases, Recall decreases")
        elif t >= 0.99:
            print(f"    -> ULTRA-CONSERVATIVE threshold. Near-zero false positives.")
            print(f"    -> FPR={m['fpr']:.4%} - virtually no legitimate URLs are blocked.")
            print(f"    -> FNR={m['fnr']:.2%} - {m['fn']:,} phishing URLs evade detection.")
            print(f"    -> Use case: Pre-filter where blocking a legitimate URL is unacceptable.")
        print()
    
    # Save results
    output_path = args.output
    if output_path is None:
        output_path = pred_path_obj.parent / "threshold_comparison_results.json"
    
    output_data = {
        'predictions_file': str(pred_path_obj),
        'total_samples': len(df),
        'label_distribution': {str(k): int(v) for k, v in label_dist.items()},
        'results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"  [OK] Results saved to: {output_path}\n")


if __name__ == "__main__":
    main()
