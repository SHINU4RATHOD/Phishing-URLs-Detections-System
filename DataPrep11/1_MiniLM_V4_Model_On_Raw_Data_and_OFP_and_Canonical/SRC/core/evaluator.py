from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
from core.config import Config


class EnhancedKPIEvaluator:
    """
    World-class KPI evaluation with multi-objective threshold optimization.
    Designed to meet strict KPIs: FPR ≤ 1%, FNR ≤ 10%, Precision ≥ 95%, Recall ≥ 95%
    """
    
    def __init__(self):
        self.evaluation_history: List[Dict] = []
    
    def evaluate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """Compute comprehensive metrics with KPI compliance check."""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.5
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        
        tpr = recall
        tnr = specificity
        fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
        for_rate = fn / (tn + fn) if (tn + fn) > 0 else 0.0
        balanced_accuracy = (tpr + tnr) / 2.0
        mcc_denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0
        
        # Individual KPI checks
        kpi_checks = {
            'accuracy_met': accuracy >= Config.TARGET_ACCURACY,
            'precision_met': precision >= Config.TARGET_PRECISION,
            'recall_met': recall >= Config.TARGET_RECALL,
            'fnr_met': fnr <= Config.MAX_FNR,
            'fpr_met': fpr <= Config.MAX_FPR,
        }
        
        kpi_compliance = all(kpi_checks.values())
        
        # Weighted KPI score (emphasizing the hardest targets)
        kpi_score = (
            0.20 * accuracy +
            0.20 * precision +
            0.20 * recall +
            0.20 * (1 - fnr) +  # Increased weight for FNR
            0.20 * (1 - fpr)   # Increased weight for FPR
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'fnr': fnr,
            'fpr': fpr,
            'specificity': specificity,
            'npv': npv,
            'tpr': tpr,
            'tnr': tnr,
            'mcc': mcc,
            'fdr': fdr,
            'for_rate': for_rate,
            'balanced_accuracy': balanced_accuracy,
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'kpi_compliance': kpi_compliance,
            'kpi_checks': kpi_checks,
            'kpi_score': kpi_score
        }
    
    def find_optimal_threshold_strict(self, y_true: np.ndarray, y_prob: np.ndarray, prioritize: str = 'balanced') -> Tuple[float, Dict]:
        """
        Find optimal threshold that satisfies STRICT KPI constraints.
        
        Strategy:
        1. First, find all thresholds satisfying FPR ≤ 1% AND FNR ≤ 10%
        2. Among valid thresholds, pick one that maximizes F1 or accuracy
        3. If no valid threshold exists, find the best compromise
        
        Args:
            y_true: Ground truth labels
            y_prob: Predicted probabilities for positive class
            prioritize: 'fpr' (minimize FPR), 'fnr' (minimize FNR), 'balanced' (maximize F1)
        
        Returns:
            optimal_threshold, metrics_at_threshold
        """
        # Clean probabilities
        valid_mask = np.isfinite(y_prob)
        if not valid_mask.all():
            print(f"[WARN] {(~valid_mask).sum()} invalid probability values detected")
            y_true = y_true[valid_mask]
            y_prob = y_prob[valid_mask]
        
        y_prob = np.clip(y_prob, 0.0, 1.0)
        
        # Search thresholds from 0.25 to 0.85 with fine granularity
        thresholds = np.arange(0.25, 0.85, 0.005)
        
        valid_thresholds = []
        all_results = []
        
        print("\n" + "=" * 70)
        print("STRICT THRESHOLD OPTIMIZATION")
        print("=" * 70)
        print(f"Constraints: FPR <= {Config.MAX_FPR:.1%}, FNR <= {Config.MAX_FNR:.1%}")
        print(f"Searching {len(thresholds)} threshold values...")
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            result = {
                'threshold': thresh,
                'fpr': fpr,
                'fnr': fnr,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'f1': f1,
                'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
            }
            all_results.append(result)
            
            # Check if this threshold satisfies BOTH constraints
            if fpr <= Config.MAX_FPR and fnr <= Config.MAX_FNR:
                valid_thresholds.append(result)
        
        # Decision logic
        if valid_thresholds:
            print(f"\n[PASS] Found {len(valid_thresholds)} valid thresholds meeting both constraints!")
            
            # Among valid thresholds, pick based on priority
            if prioritize == 'fpr':
                best = min(valid_thresholds, key=lambda x: (x['fpr'], -x['f1']))
            elif prioritize == 'fnr':
                best = min(valid_thresholds, key=lambda x: (x['fnr'], -x['f1']))
            else:  # balanced
                best = max(valid_thresholds, key=lambda x: x['f1'])
            
            print(f"\nOptimal Threshold: {best['threshold']:.3f}")
            print(f"  FPR:       {best['fpr']:.4f} (target <= {Config.MAX_FPR})")
            print(f"  FNR:       {best['fnr']:.4f} (target <= {Config.MAX_FNR})")
            print(f"  Precision: {best['precision']:.4f}")
            print(f"  Recall:    {best['recall']:.4f}")
            print(f"  F1:        {best['f1']:.4f}")
            print(f"  Accuracy:  {best['accuracy']:.4f}")
            
        else:
            print(f"\n[WARN] No threshold satisfies both FPR <= {Config.MAX_FPR:.1%} AND FNR <= {Config.MAX_FNR:.1%}")
            print("Finding best compromise...")
            
            # Find threshold that minimizes combined violation
            def violation_score(r):
                fpr_violation = max(0, r['fpr'] - Config.MAX_FPR)
                fnr_violation = max(0, r['fnr'] - Config.MAX_FNR)
                return fpr_violation + fnr_violation - 0.1 * r['f1']  # Small bonus for F1
            
            best = min(all_results, key=violation_score)
            
            print(f"\nBest Compromise Threshold: {best['threshold']:.3f}")
            print(f"  FPR:       {best['fpr']:.4f} {'[OK]' if best['fpr'] <= Config.MAX_FPR else '[FAIL]'}")
            print(f"  FNR:       {best['fnr']:.4f} {'[OK]' if best['fnr'] <= Config.MAX_FNR else '[FAIL]'}")
        
        print("=" * 70)
        
        return best['threshold'], best
    
    def analyze_threshold_sensitivity(self, y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
        """Generate threshold sensitivity analysis table."""
        thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        results = []
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            results.append({
                'Threshold': thresh,
                'FPR': fp / (fp + tn),
                'FNR': fn / (fn + tp),
                'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'Accuracy': (tp + tn) / (tp + tn + fp + fn),
                'FPR_OK': '[OK]' if fp / (fp + tn) <= Config.MAX_FPR else '[FAIL]',
                'FNR_OK': '[OK]' if fn / (fn + tp) <= Config.MAX_FNR else '[FAIL]',
            })
        
        return pd.DataFrame(results)
