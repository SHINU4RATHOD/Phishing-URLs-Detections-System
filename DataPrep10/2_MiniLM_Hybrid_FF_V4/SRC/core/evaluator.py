from typing import Dict, Tuple, Optional, List, Any
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from core.config import Config


class SamsungDecisionEngine:
    """
    Production-grade Risk Engine bridging smooth ML probabilities with hard cybersecurity constraints.
    Replaces static thresholds with dynamic Risk Scores and calibrated probability boundaries.
    """
    
    HIGH_RISK_CATS = [
        'Credential_Harvesting_Form_URL', 'IsSuspiciousFileType', 
        'IsObfuscatedURL', 'TypoSquatting_URL', 'Compromised_CMS_URL'
    ]
    SAFE_INFRA_CATS = [
        'IsLanguageSpecific', 'Anchor_Fragment_Based_URL', 'Cloud_Hosting_Abuse_URL'
    ]
    
    @classmethod
    def decide(cls, log_odds: np.ndarray, severities: np.ndarray, flags_counts: np.ndarray, 
               categories: np.ndarray, base_threshold: float = 0.5, lambda_val: float = 0.5) -> np.ndarray:
        """
        Accepts raw log-odds already calibrated by Temperature.
        Operates entirely in logit space.
        
        Args:
            lambda_val: Logit boost strength for high-risk categories.
        """
        clamped_logits = np.clip(log_odds, -20.0, 20.0)
        
        # 1. Normalize metadata
        norm_sev = np.clip(severities / 10.0, 0.0, 1.0)
        norm_flags = np.clip(flags_counts / 10.0, 0.0, 1.0)
        
        # 2. Compute Structural Risk Score
        engine_risk_score = 0.60 * (norm_sev ** 1.5) + 0.40 * (norm_flags ** 1.3)
        
        # 3. Category-Aware Logit Adjustment
        adjusted_logits = clamped_logits.copy()
        
        high_risk_mask = np.isin(categories, cls.HIGH_RISK_CATS)
        safe_infra_mask = np.isin(categories, cls.SAFE_INFRA_CATS)
        ambiguous_mask = ~(high_risk_mask | safe_infra_mask)
        
        # Boost logits for high-risk categories
        adjusted_logits[high_risk_mask] += lambda_val * engine_risk_score[high_risk_mask]
        
        # Convert to probability (single sigmoid pass)
        adjusted_probs = 1.0 / (1.0 + np.exp(-adjusted_logits))
        
        # 4. Contextual Threshold Offsets (conservative to protect FPR)
        local_thresholds = np.full_like(adjusted_probs, base_threshold)
        local_thresholds[high_risk_mask] = np.maximum(base_threshold * 0.92, 0.40)
        local_thresholds[safe_infra_mask] = np.minimum(base_threshold + 0.15, 0.90)
        
        clean_mask = ambiguous_mask & (flags_counts == 0) & (severities <= 1.0)
        local_thresholds[clean_mask] = np.minimum(base_threshold + 0.15, 0.85)

        # Vectorized final decision
        final_preds = (adjusted_probs >= local_thresholds).astype(int)
        
        return final_preds


class EnhancedKPIEvaluator:
    """
    Enhanced KPI evaluation with multi-objective threshold optimization.
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
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        tpr = recall
        tnr = specificity
        fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
        for_rate = fn / (tn + fn) if (tn + fn) > 0 else 0.0
        balanced_accuracy = (tpr + tnr) / 2.0
        mcc_denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0
        
        kpi_checks = {
            'accuracy_met': accuracy >= Config.TARGET_ACCURACY,
            'precision_met': precision >= Config.TARGET_PRECISION,
            'recall_met': recall >= Config.TARGET_RECALL,
            'fnr_met': fnr <= Config.MAX_FNR,
            'fpr_met': fpr <= Config.MAX_FPR,
        }
        
        kpi_compliance = all(kpi_checks.values())
        
        # Weighted KPI score
        kpi_score = (
            0.10 * accuracy +
            0.20 * precision +
            0.15 * recall +
            0.20 * (1 - fnr) +  
            0.35 * (1 - fpr)   
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
    
    def find_optimal_lambda_threshold_joint(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                           metadata: Optional[Dict] = None, 
                                           log_odds: Optional[np.ndarray] = None) -> Tuple[float, float, Dict]:
        """
        Joint (λ, threshold) grid search.
        
        Returns:
            (optimal_lambda, optimal_threshold, best_metrics)
        """
        if metadata is None or log_odds is None:
            # Fallback: simple threshold-only search
            best_f1, best_t = 0.0, 0.5
            for t in np.arange(0.30, 0.85, 0.005):
                preds = (y_prob >= t).astype(int)
                cm = confusion_matrix(y_true, preds, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            return 0.0, best_t, {'lambda': 0.0, 'threshold': best_t, 'f1': best_f1, 'fpr': 0.0, 'fnr': 0.0, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0}
        
        lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]
        thresholds = np.arange(0.30, 0.85, 0.005)
        
        valid_configs = []
        best_compromise = None
        best_compromise_score = float('inf')
        
        print("\n" + "=" * 70)
        print("JOINT (LAMBDA, THRESHOLD) GRID SEARCH")
        print("=" * 70)
        print(f"Constraints: FPR <= {Config.MAX_FPR:.1%}, FNR <= {Config.MAX_FNR:.1%}")
        print(f"Searching {len(lambda_values)} lambda values x {len(thresholds)} thresholds = {len(lambda_values)*len(thresholds)} combinations...")
        
        for lam in lambda_values:
            for thresh in thresholds:
                y_pred = SamsungDecisionEngine.decide(
                    log_odds=log_odds,
                    severities=metadata['severities'],
                    flags_counts=metadata['flags_counts'],
                    categories=metadata['categories'],
                    base_threshold=thresh,
                    lambda_val=lam
                )
                
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel()
                
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                result = {
                    'lambda': lam, 'threshold': thresh,
                    'fpr': fpr, 'fnr': fnr, 'precision': precision,
                    'recall': recall, 'accuracy': accuracy, 'f1': f1,
                    'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
                }
                
                fpr_viol = max(0, fpr - Config.MAX_FPR)
                fnr_viol = max(0, fnr - Config.MAX_FNR)
                viol_score = fpr_viol + fnr_viol - 0.05 * f1
                if viol_score < best_compromise_score:
                    best_compromise_score = viol_score
                    best_compromise = result
                
                if fpr <= Config.MAX_FPR and fnr <= Config.MAX_FNR:
                    valid_configs.append(result)
        
        if valid_configs:
            print(f"\n[OK] Found {len(valid_configs)} valid (lambda, threshold) pairs meeting BOTH constraints!")
            best = max(valid_configs, key=lambda x: x['f1'])
            print(f"\n[OPTIMAL OPERATING POINT]:")
            print(f"  lambda:      {best['lambda']:.2f}")
            print(f"  Threshold:   {best['threshold']:.3f}")
            print(f"  FPR:         {best['fpr']:.4f} (target <= {Config.MAX_FPR}) [PASS]")
            print(f"  FNR:         {best['fnr']:.4f} (target <= {Config.MAX_FNR}) [PASS]")
            print(f"  Precision:   {best['precision']:.4f}")
            print(f"  Recall:      {best['recall']:.4f}")
            print(f"  F1:          {best['f1']:.4f}")
            print(f"  Accuracy:    {best['accuracy']:.4f}")
        else:
            print(f"\n[WARN] No (lambda, threshold) pair satisfies BOTH FPR <= {Config.MAX_FPR:.1%} AND FNR <= {Config.MAX_FNR:.1%}")
            print("Using best compromise...")
            best = best_compromise
            print(f"\n[BEST COMPROMISE]:")
            print(f"  lambda:      {best['lambda']:.2f}")
            print(f"  Threshold:   {best['threshold']:.3f}")
            print(f"  FPR:         {best['fpr']:.4f} {'[PASS]' if best['fpr'] <= Config.MAX_FPR else '[FAIL]'}")
            print(f"  FNR:         {best['fnr']:.4f} {'[PASS]' if best['fnr'] <= Config.MAX_FNR else '[FAIL]'}")
            print(f"  F1:          {best['f1']:.4f}")
        
        print("=" * 70)
        return best['lambda'], best['threshold'], best
    
    def analyze_threshold_sensitivity(self, y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
        """Generate threshold sensitivity analysis table."""
        thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        results = []
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            
            results.append({
                'Threshold': thresh,
                'FPR': fp / (fp + tn),
                'FNR': fn / (fn + tp),
                'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'Accuracy': (tp + tn) / (tp + tn + fp + fn),
                'FPR_OK': '✓' if fp / (fp + tn) <= Config.MAX_FPR else '✗',
                'FNR_OK': '✓' if fn / (fn + tp) <= Config.MAX_FNR else '✗',
            })
        
        return pd.DataFrame(results)
