import os
import json
import random
import warnings
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from collections import Counter  
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast

from transformers import AutoTokenizer, AutoModel, AutoConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import onnx

# ONNX Quantization Setup
ONNX_QUANTIZATION_AVAILABLE = False
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_QUANTIZATION_AVAILABLE = True
    print("[OK] ONNX quantization available")
except ImportError:
    print("[WARN] ONNX quantization unavailable - install: pip install onnxruntime")

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Reproducibility
    SEED: int = 42
    
    # # Data Paths - OFP (Original Format Preprocessed) splits
    TRAIN_CSV: str = "/home/hp/SHINU RATHOD/URLsPhishDetect-with-LLMS/2_Model _Preprocessed_data/Data_Preprocessing/data_prep8/preprocess_urls_output_103_dataset/raw_orig_train.csv"
    VAL_CSV: str = "/home/hp/SHINU RATHOD/URLsPhishDetect-with-LLMS/2_Model _Preprocessed_data/Data_Preprocessing/data_prep8/preprocess_urls_output_103_dataset/raw_orig_val.csv"
    TEST_CSV: str = "/home/hp/SHINU RATHOD/URLsPhishDetect-with-LLMS/2_Model _Preprocessed_data/Data_Preprocessing/data_prep8/preprocess_urls_output_103_dataset/raw_orig_test.csv"

    # Data Paths - OFP (Original Format Preprocessed) splits
    # TRAIN_CSV: str = "D:/IIT ROPAR/phishing URL Detection/01_Research Tracker/2_Model_Building/PhishURLDetect-with-LLMS/2_Model _Preprocessed_data/Data_Preprocessing/data_prep7/preprocess_urls_output/urls_ofp_train.csv"
    # VAL_CSV: str = "D:/IIT ROPAR/phishing URL Detection/01_Research Tracker/2_Model_Building/PhishURLDetect-with-LLMS/2_Model _Preprocessed_data/Data_Preprocessing/data_prep7/preprocess_urls_output/urls_ofp_val.csv"
    # TEST_CSV: str = "D:/IIT ROPAR/phishing URL Detection/01_Research Tracker/2_Model_Building/PhishURLDetect-with-LLMS/2_Model _Preprocessed_data/Data_Preprocessing/data_prep7/preprocess_urls_output/urls_ofp_test.csv"
    
    
    # ========================================================================
    # MODEL ARCHITECTURE - MiniLM-L12-H384 (OPTIMIZED FOR URL CLASSIFICATION)
    # ========================================================================
    MODEL_NAME: str = "microsoft/MiniLM-L12-H384-uncased"  # Compact & efficient
    MAX_LEN: int = 128          # Optimized: OFP canonical URLs avg ~80 chars
    NUM_CLASSES: int = 2
    DROPOUT: float = 0.15       # Reduced slightly - 26.5M samples = almost zero overfit risk
    CLASSIFIER_DIMS: List[int] = [384, 512, 256, 128, 64]  # Engineered bottleneck: distills 384-dim semantics before binary decision
    
    # ========================================================================
    # TRAINING CONFIG - OPTIMIZED FOR 26.5M SAMPLES
    # ========================================================================
    BATCH_SIZE: int = 128       # User requested (larger for stable gradients)
    NUM_EPOCHS: int = 3        # 26.5M x 3 = 80M+ samples seen (sufficient with higher LR)
    WEIGHT_DECAY: float = 0.01  # Relaxed to let weights grow given the massive dataset
    PATIENCE: int = 3           # Matches 3-epoch budget
    GRAD_ACCUM_STEPS: int = 4   # Effective batch = 256 (optimal for classification)
    GRAD_CLIP_NORM: float = 1.0 # Standard - 0.5 was too aggressive
    
    # ========================================================================
    # LEARNING RATE SCHEDULE (Cosine with Warmup) - RESEARCH OPTIMIZED
    # ========================================================================
    LR: float = 1e-4            # Boosted for LoRA - necessary to converge within 3 epochs on massive data
    LR_WARMUP_RATIO: float = 0.03  # 3% warmup - more learning budget with fewer epochs
    LR_MIN_RATIO: float = 0.001    # Lower minimum for better convergence
    
    # ========================================================================
    # LoRA CONFIG - OPTIMIZED FOR CLASSIFICATION
    # ========================================================================
    LORA_R: int = 32            # Increased capacity to memorize hard edge-cases for strict KPIs
    LORA_ALPHA: int = 64        # Best practice: 2 × Rank
    LORA_DROPOUT: float = 0.05  # Minimal dropout due to massive dataset size
    LORA_TARGET_MODULES: List[str] = ["query", "key", "value",  "dense", "output.dense"]

    # ========================================================================
    # FOCAL LOSS - SOLE CLASS BALANCER (no weighted sampling due to 2^24 limit)
    # Ratio: 73.2% benign / 26.8% phishing = 2.72:1 imbalance
    # ========================================================================
    FOCAL_GAMMA_POS: float = 2.0
    FOCAL_GAMMA_NEG: float = 4.0    # Strict penalty on hard FP/FN errors (crucial for 1% FPR / 10% FNR)
    FOCAL_ALPHA: List[float] = [0.28, 0.72]  # Mathematical exact inverse of 71.6% Benign / 28.4% Phishing
    LABEL_SMOOTHING: float = 0.05  # Increased for better probability calibration at extreme thresholds
    
    # ========================================================================
    # CLASS BALANCING
    # ========================================================================
    USE_STRATIFIED_KFOLD: bool = True
    KFOLD_SPLITS: int = 5
    USE_WEIGHTED_SAMPLING: bool = False  # Disabled: PyTorch limit 2^24 samples; focal loss α handles imbalance
    
    # OPTIMIZATION
    PRUNING_RATIO: float = 0.0
    USE_AMP: bool = True
    
    # Export
    EXPORT_ONNX: bool = True
    EXPORT_QUANTIZED: bool = True
    ONNX_OPSET: int = 14
    
    # KPI TARGETS (STRICT)
    TARGET_ACCURACY: float = 0.98
    TARGET_PRECISION: float = 0.95
    TARGET_RECALL: float = 0.95
    MAX_FNR: float = 0.10
    MAX_FPR: float = 0.01
    MAX_MODEL_SIZE_MB: float = 40.0  # MiniLM target: <40MB
    
    # Hardware - OPTIMIZED FOR LARGE DATASET
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS: int = 12       # Set to 0 to avoid Windows multiprocessing DataLoader crashes
    PIN_MEMORY: bool = True
    PREFETCH_FACTOR: int = 4    # Increased from 2 for GPU utilization 
    
    # Paths
    SAVE_ROOT: Optional[Path] = None
    CHECKPOINT_DIR: Optional[Path] = None
    
    @classmethod
    def setup_paths(cls) -> None:
        """Initialize output directories."""
        cls.SAVE_ROOT = Path(f"saved_models/MiniLM_raw_orig4")
        cls.CHECKPOINT_DIR = cls.SAVE_ROOT / "checkpoints"
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.SAVE_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Save directory: {cls.SAVE_ROOT}")
    
    @classmethod
    def setup_reproducibility(cls) -> None:
        """Set random seeds."""
        torch.manual_seed(cls.SEED)
        np.random.seed(cls.SEED)
        random.seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
class FocalLoss(nn.Module):
    """Focal Loss with numerical stability."""
    
    def __init__(self):
        super().__init__()
        self.gamma_pos = Config.FOCAL_GAMMA_POS
        self.gamma_neg = Config.FOCAL_GAMMA_NEG
        self.label_smoothing = Config.LABEL_SMOOTHING
        
        if Config.FOCAL_ALPHA:
            self.register_buffer('alpha_tensor', torch.tensor(Config.FOCAL_ALPHA, dtype=torch.float))
        else:
            self.alpha_tensor = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, min=-10, max=10)
        ce_loss = nn.functional.cross_entropy( logits, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss).clamp(min=1e-7, max=1.0)
        
        gamma = torch.where(targets == 1, self.gamma_pos, self.gamma_neg)
        
        if self.alpha_tensor is not None:
            alpha_t = self.alpha_tensor.to(targets.device)[targets]
            focal_loss = alpha_t * ((1 - pt) ** gamma) * ce_loss
        else:
            focal_loss = ((1 - pt) ** gamma) * ce_loss
        
        loss = focal_loss.mean()
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("[WARN] NaN/Inf detected in loss, using CE fallback")
            return nn.functional.cross_entropy(logits, targets)
        
        return loss


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
class MiniLMURLClassifier(nn.Module):
    """MiniLM-L12-H384 classifier optimized for URL phishing detection."""
    
    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.encoder = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)
        self.hidden_size = self.config.hidden_size  # 384 for MiniLM-L12-H384
        
        self.dropouts = nn.ModuleList([nn.Dropout(Config.DROPOUT) for _ in range(5)])
        
        # Deeper classifier head with LayerNorm (optimized for MiniLM)
        layers = []
        in_dim = self.hidden_size
        
        for out_dim in Config.CLASSIFIER_DIMS:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(Config.DROPOUT)
            ])
            in_dim = out_dim
        
        layers.append(nn.Linear(in_dim, Config.NUM_CLASSES))
        self.classifier = nn.Sequential(*layers)
        self._init_classifier_weights()
    
    def _init_classifier_weights(self) -> None:
        """Xavier initialization with small std for stability."""
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        pooled_output = self.mean_pooling(outputs, attention_mask)
        
        logits = None
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logits = self.classifier(dropout(pooled_output))
            else:
                logits += self.classifier(dropout(pooled_output))
        logits = logits / len(self.dropouts)
        
        # Stability check
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("[WARN] NaN/Inf detected in logits")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return logits


def apply_structured_pruning(model: nn.Module, amount: float = Config.PRUNING_RATIO) -> None:
    """Pruning disabled for MiniLM stability."""
    if amount <= 0.0:
        print("Pruning disabled (MiniLM already compact)")
        return

from torchinfo import summary
def save_model_summary(model: nn.Module, input_size: Tuple[int, int], save_path: str = "model_summery.txt") -> None:
    '''
    Save comprehensive model summary to a text file.
    
    This simplified version doesn't use torchinfo, making it more reliable
    for models with multiple inputs like MiniLM.
    '''
    try:
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        # Calculate model size in MB
        param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        
        # Build summary string
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("MODEL SUMMARY: MiniLM URL Classifier")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        summary_lines.append(f"Model Architecture: {model.__class__.__name__}")
        summary_lines.append(f"Input Size (batch, seq_len): {input_size}")
        summary_lines.append("")
        summary_lines.append("-" * 80)
        summary_lines.append("PARAMETER STATISTICS")
        summary_lines.append("-" * 80)
        summary_lines.append(f"Total Parameters:         {total_params:,}")
        summary_lines.append(f"Trainable Parameters:     {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        summary_lines.append(f"Non-trainable Parameters: {non_trainable_params:,} ({non_trainable_params/total_params*100:.2f}%)")
        summary_lines.append(f"Model Size:               {size_mb:.2f} MB")
        summary_lines.append("")
        summary_lines.append("-" * 80)
        summary_lines.append("LAYER-WISE BREAKDOWN")
        summary_lines.append("-" * 80)
        summary_lines.append(f"{'Layer Name':<50} {'Parameters':>15} {'Trainable':>12}")
        summary_lines.append("-" * 80)
        
        for name, param in model.named_parameters():
            trainable = "Yes" if param.requires_grad else "No"
            summary_lines.append(f"{name:<50} {param.numel():>15,} {trainable:>12}")
        
        summary_lines.append("=" * 80)
        summary_str = "\n".join(summary_lines)
        
        # Write to file
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(summary_str)
        
        print(f"[OK] Model summary saved to {save_path}")
        print(f"  Total params: {total_params:,} | Trainable: {trainable_params:,} | Size: {size_mb:.2f} MB")
        
    except Exception as e:
        print(f"✗ Failed to save model summary: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# DATASET
# ============================================================================
class URLDataset(Dataset):
    """PyTorch Dataset for URL classification."""
    
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.tokenizer = tokenizer
        self.urls = df['input'].astype(str).tolist()
        self.labels = df['label'].astype(int).tolist()
        
        print(f"Dataset: {len(self.urls):,} samples")
        label_dist = pd.Series(self.labels).value_counts().to_dict()
        print(f"Label distribution: {label_dist}")
    
    def __len__(self) -> int:
        return len(self.urls)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        url = self.urls[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer( url, add_special_tokens=True, max_length=Config.MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'url': url
        }
    
def create_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    # Count samples per class
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    # Calculate inverse frequency weights
    class_weights = {
        class_id: total_samples / count 
        for class_id, count in class_counts.items()
    }
    
    # Assign weight to each sample based on its class
    sample_weights = [class_weights[label] for label in labels]
    
    # Create sampler
    sampler = WeightedRandomSampler( weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    print(f"\n{'='*60}")
    print("WEIGHTED SAMPLING ACTIVATED")
    print(f"{'='*60}")
    print(f"Class distribution:")
    for class_id, count in sorted(class_counts.items()):
        percentage = (count / total_samples) * 100
        weight = class_weights[class_id]
        label_name = "Benign" if class_id == 0 else "Phishing"
        print(f"  {label_name:10} ({class_id}): {count:,} samples ({percentage:.2f}%) - weight: {weight:.4f}")
    print(f"{'='*60}\n")
    
    return sampler


# ============================================================================
# ENHANCED KPI EVALUATOR - STRICT THRESHOLD OPTIMIZATION
# ============================================================================
class EnhancedKPIEvaluator:
    """
    World-class KPI evaluation with multi-objective threshold optimization.
    Designed to meet strict KPIs: FPR ≤ 1%, FNR ≤ 10%, Precision ≥ 95%, Recall ≥ 95%
    """
    
    def __init__(self):
        self.evaluation_history: List[Dict] = []
    
    def evaluate_metrics(self,  y_true: np.ndarray,  y_pred: np.ndarray,  y_prob: np.ndarray) -> Dict[str, Any]:
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
    
    def find_optimal_threshold_strict(self,  y_true: np.ndarray,  y_prob: np.ndarray, prioritize: str = 'balanced') -> Tuple[float, Dict]:
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
        
        # Search thresholds from 0.30 to 0.85 with fine granularity
        thresholds = np.arange(0.25, 0.85, 0.005)
        
        valid_thresholds = []
        all_results = []
        
        print("\n" + "=" * 70)
        print("STRICT THRESHOLD OPTIMIZATION")
        print("=" * 70)
        print(f"Constraints: FPR ≤ {Config.MAX_FPR:.1%}, FNR ≤ {Config.MAX_FNR:.1%}")
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
            print(f"  FPR:       {best['fpr']:.4f} (target ≤ {Config.MAX_FPR})")
            print(f"  FNR:       {best['fnr']:.4f} (target ≤ {Config.MAX_FNR})")
            print(f"  Precision: {best['precision']:.4f}")
            print(f"  Recall:    {best['recall']:.4f}")
            print(f"  F1:        {best['f1']:.4f}")
            print(f"  Accuracy:  {best['accuracy']:.4f}")
            
        else:
            print(f"\n[WARN] No threshold satisfies both FPR ≤ {Config.MAX_FPR:.1%} AND FNR ≤ {Config.MAX_FNR:.1%}")
            print("Finding best compromise...")
            
            # Find threshold that minimizes combined violation
            def violation_score(r):
                fpr_violation = max(0, r['fpr'] - Config.MAX_FPR)
                fnr_violation = max(0, r['fnr'] - Config.MAX_FNR)
                return fpr_violation + fnr_violation - 0.1 * r['f1']  # Small bonus for F1
            
            best = min(all_results, key=violation_score)
            
            print(f"\nBest Compromise Threshold: {best['threshold']:.3f}")
            print(f"  FPR:       {best['fpr']:.4f} {'[OK]' if best['fpr'] <= Config.MAX_FPR else '✗'}")
            print(f"  FNR:       {best['fnr']:.4f} {'[OK]' if best['fnr'] <= Config.MAX_FNR else '✗'}")
        
        print("=" * 70)
        
        return best['threshold'], best
    
    def analyze_threshold_sensitivity(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray
    ) -> pd.DataFrame:
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
                'FPR_OK': '[OK]' if fp / (fp + tn) <= Config.MAX_FPR else '✗',
                'FNR_OK': '[OK]' if fn / (fn + tp) <= Config.MAX_FNR else '✗',
            })
        
        return pd.DataFrame(results)


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================
class CheckpointManager:
    """Handles model checkpointing with resume support."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint( self, model: nn.Module, optimizer: optim.Optimizer, scheduler: Optional[Any], scaler: Optional[GradScaler], epoch: int, metrics: Dict, threshold: float, best_kpi_score: float, training_history: Dict) -> Path:
        """Save training checkpoint with full state."""
        config_dict = self._serialize_config()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if Config.USE_AMP else None,
            'metrics': metrics,
            'threshold': threshold,
            'best_kpi_score': best_kpi_score,
            'training_history': training_history,
            'config': config_dict
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        temp_path = checkpoint_path.with_suffix('.pt.tmp')
        
        try:
            torch.save(checkpoint, temp_path)
            temp_path.rename(checkpoint_path)
            print(f"[OK] Checkpoint saved: {checkpoint_path.name}")
        except Exception as e:
            print(f"✗ Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
        
        return checkpoint_path
    
    def load_checkpoint( self, checkpoint_path: Path, model: nn.Module, optimizer: Optional[optim.Optimizer] = None, scheduler: Optional[Any] = None, scaler: Optional[GradScaler] = None) -> Tuple[int, Dict, float, Dict]:
        """Load checkpoint and restore full training state."""
        try:
            print(f"\n{'='*60}")
            print(f"RESUMING FROM CHECKPOINT")
            print(f"{'='*60}")
            
            # weights_only=False to load full training state (PyTorch 2.6+ default changed)
            checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=False)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.to(Config.DEVICE)
            
            # Load optimizer state
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"[OK] Optimizer state restored")
            
            # Load scheduler state
            if scheduler and checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"[OK] Scheduler state restored")
            
            # Load scaler state
            if scaler and checkpoint.get('scaler_state_dict'):
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print(f"[OK] Scaler state restored")
            
            # Extract training state
            start_epoch = checkpoint.get('epoch', 0) + 1
            metrics = checkpoint.get('metrics', {})
            best_kpi_score = checkpoint.get('best_kpi_score', 0.0)
            training_history = checkpoint.get('training_history', {})
            
            print(f"[OK] Model state restored")
            print(f"[OK] Resuming from epoch {start_epoch}")
            print(f"[OK] Best KPI score: {best_kpi_score:.4f}")
            print(f"{'='*60}\n")
            
            return start_epoch, metrics, best_kpi_score, training_history
        
        except Exception as e:
            print(f"[WARN] Checkpoint load failed ({checkpoint_path.name}): {e}")
            print(f"[WARN] Keeping the checkpoint file for inspection. Starting fresh this run.")
            return 1, {}, 0.0, {}
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find most recent valid checkpoint (non-destructive)."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        latest_valid = None
        for ckpt in reversed(checkpoints):
            try:
                torch.load(ckpt, map_location='cpu', weights_only=False)
                latest_valid = ckpt
                break
            except Exception as e:
                print(f"[WARN] Corrupted checkpoint: {ckpt.name} ({e})")
                continue
        
        return latest_valid
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 3):
        """Keep only the last N checkpoints to save disk space."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        if len(checkpoints) > keep_last_n:
            for ckpt in checkpoints[:-keep_last_n]:
                try:
                    ckpt.unlink()
                    print(f"🗑 Cleaned up old checkpoint: {ckpt.name}")
                except Exception as e:
                    print(f"[WARN] Failed to delete {ckpt.name}: {e}")
    
    @staticmethod
    def _serialize_config() -> Dict:
        """Serialize Config to dict."""
        config_dict = {}
        for key, value in vars(Config).items():
            if key.startswith('_') or callable(value):
                continue
            if isinstance(value, Path):
                config_dict[key] = str(value)
            elif isinstance(value, (int, float, str, bool, list, dict, type(None))):
                config_dict[key] = value
            else:
                config_dict[key] = str(value)
        return config_dict


# ============================================================================
# ARTIFACT SAVER
# ============================================================================
class ArtifactSaver:
    """Saves training artifacts, plots, and metrics."""
    
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    def save_history( self, train_losses: List[float], val_losses: List[float], train_accs: List[float], val_accs: List[float]) -> None:
        """Save training history CSV and plots."""
        history_df = pd.DataFrame({
            'epoch': range(1, len(train_losses) + 1),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_acc': train_accs,
            'val_acc': val_accs
        })
        history_df.to_csv(self.run_dir / 'training_history.csv', index=False)
        print(f"[OK] Training history saved")
        
        self._plot_curves(history_df)
    
    def _plot_curves(self, df: pd.DataFrame) -> None:
        """Generate loss and accuracy plots."""
        # Loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(df['epoch'], df['train_loss'], label='Train', linewidth=2, marker='o')
        plt.plot(df['epoch'], df['val_loss'], label='Validation', linewidth=2, marker='s')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.run_dir / 'loss_curves.png', dpi=300)
        plt.close()
        
        # Accuracy curves
        plt.figure(figsize=(10, 5))
        plt.plot(df['epoch'], df['train_acc'], label='Train', linewidth=2, marker='o')
        plt.plot(df['epoch'], df['val_acc'], label='Validation', linewidth=2, marker='s')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.run_dir / 'accuracy_curves.png', dpi=300)
        plt.close()
        
        print(f"[OK] Training plots saved")
    
    def save_test_metrics(self, metrics: Dict, threshold: float) -> None:
        """Save test metrics to CSV."""
        metrics_copy = metrics.copy()
        metrics_copy['threshold'] = threshold
        pd.DataFrame([metrics_copy]).to_csv(self.run_dir / 'test_metrics.csv', index=False)
        print(f"[OK] Test metrics saved")
    
    def save_test_plots( self, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> None:
        """Generate test set visualizations."""
        y_pred = (y_prob >= threshold).astype(int)
        
        self._plot_confusion_matrix(y_true, y_pred, threshold)
        self._plot_roc_curve(y_true, y_prob)
        self._plot_pr_curve(y_true, y_prob)
        print(f"[OK] Test plots saved")
    
    def _plot_confusion_matrix( self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> None:
        """Plot confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        cm_percent = cm / total * 100
        
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        plt.figure(figsize=(8, 6))
        sns.heatmap( cm_percent, annot=False, fmt='.1f', cmap='Blues', xticklabels=['Benign', 'Malicious'], yticklabels=['Benign', 'Malicious'], cbar_kws={'label': 'Percentage'})
        
        # Annotate cells
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                percent = cm_percent[i, j]
                plt.text(
                    j + 0.5, i + 0.5,
                    f'{percent:.1f}%\n({count:,})',
                    ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color='white' if percent > 50 else 'black'
                )
        
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=14, fontweight='bold')
        plt.title( f'Test Confusion Matrix (Threshold={threshold:.3f})\n' f'FNR={fnr:.2%} | FPR={fpr:.2%}', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.run_dir / 'confusion_matrix_test.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}', linewidth=2.5, color='blue')
        plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=2, label='Random')
        plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        plt.title('ROC Curve - Test Set', fontsize=16, fontweight='bold', pad=20)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.run_dir / 'roc_test.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pr_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Plot precision-recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2.5, color='green')
        plt.xlabel('Recall', fontsize=14, fontweight='bold')
        plt.ylabel('Precision', fontsize=14, fontweight='bold')
        plt.title('Precision-Recall Curve - Test Set', fontsize=16, fontweight='bold', pad=20)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.run_dir / 'pr_curve_test.png', dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================================
# MODEL EXPORTER
# ============================================================================
class ModelExporter:
    """Handles model export to ONNX with quantization."""
    
    @staticmethod
    def merge_lora_and_export(model: nn.Module, tokenizer, save_dir: Path) -> Tuple[nn.Module, float]:
        """Merge LoRA adapters and save production model."""
        try:
            print("Merging LoRA adapters...")
            adapter_path = save_dir / "lora_adapter"
            base_model = MiniLMURLClassifier()
            merged_model = PeftModel.from_pretrained(base_model, str(adapter_path))
            merged_model = merged_model.merge_and_unload()
            merged_model = merged_model.to(Config.DEVICE).eval()
            
            # Save merged model
            merged_path = save_dir / "model_merged_full.pt"
            torch.save(merged_model, merged_path)
            merged_size = os.path.getsize(merged_path) / (1024 * 1024)
            
            print(f"[OK] Merged model: {merged_size:.2f} MB")
            
            # Save state dict
            torch.save(merged_model.state_dict(), save_dir / "model_merged_state_dict.pt")

            # Save model summary
            summary_path = save_dir / "model_summery.txt"
            save_model_summary(merged_model, input_size=(1, Config.MAX_LEN), save_path=str(summary_path))

            
            return merged_model, merged_size
        
        except Exception as e:
            print(f"[WARN] LoRA merge failed: {e}")
            return model, 0.0
    
    @staticmethod
    def export_onnx( model: nn.Module, save_dir: Path) -> Optional[float]:
        """Export model to ONNX format with quantization."""
        if not Config.EXPORT_ONNX:
            return None
        
        try:
            print("Exporting to ONNX...")
            model.eval()
            device = next(model.parameters()).device
            
            dummy_input = {
                'input_ids': torch.randint(0, 30522, (1, Config.MAX_LEN), dtype=torch.long).to(device),
                'attention_mask': torch.ones(1, Config.MAX_LEN, dtype=torch.long).to(device)
            }
            
            onnx_path = save_dir / "model.onnx"
            torch.onnx.export(
                model,
                (dummy_input['input_ids'], dummy_input['attention_mask']),
                str(onnx_path),
                opset_version=Config.ONNX_OPSET,
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch', 1: 'sequence'},
                    'attention_mask': {0: 'batch', 1: 'sequence'},
                    'logits': {0: 'batch'}
                },
                verbose=False
            )
            
            onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"[OK] ONNX model: {onnx_size:.2f} MB")
            
            # Quantize ONNX
            if Config.EXPORT_QUANTIZED and ONNX_QUANTIZATION_AVAILABLE:
                quant_size = ModelExporter._quantize_onnx(onnx_path, save_dir)
                return quant_size if quant_size else onnx_size
            
            return onnx_size
        
        except Exception as e:
            print(f"✗ ONNX export failed: {e}")
            return None
    
    @staticmethod
    def _quantize_onnx(onnx_path: Path, save_dir: Path) -> Optional[float]:
        """Quantize ONNX model to 8-bit."""
        try:
            print("Quantizing ONNX model...")
            quant_path = save_dir / "model_quant_8bit.onnx"
            
            quantize_dynamic(
                str(onnx_path),
                str(quant_path),
                weight_type=QuantType.QUInt8
            )
            
            quant_size = os.path.getsize(quant_path) / (1024 * 1024)
            original_size = os.path.getsize(onnx_path) / (1024 * 1024)
            reduction = ((original_size - quant_size) / original_size) * 100
            
            print(f"[OK] Quantized ONNX: {quant_size:.2f} MB ({reduction:.1f}% reduction)")
            
            if quant_size <= Config.MAX_MODEL_SIZE_MB:
                print(f"[PASS] Quantized model meets {Config.MAX_MODEL_SIZE_MB}MB target!")
            else:
                print(f"[WARN] Quantized model {quant_size:.2f}MB exceeds {Config.MAX_MODEL_SIZE_MB}MB target")
            
            return quant_size
        
        except Exception as e:
            print(f"[WARN] ONNX quantization failed: {e}")
            return None


# ============================================================================
# TRAINER
# ============================================================================
class PhishingDetectionTrainer:
    """Main training orchestrator."""
    
    def __init__(self):
        Config.setup_reproducibility()
        Config.setup_paths()
        
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.checkpoint_manager = CheckpointManager(Config.CHECKPOINT_DIR)
        self.kpi_evaluator = EnhancedKPIEvaluator()
        
        self.training_history = {
            'train_losses': [], 'val_losses': [],
            'train_accs': [], 'val_accs': [],
            'kpi_scores': [], 'thresholds': []
        }
    
    def load_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load datasets as DataFrames to allow dynamic Stratified K-Fold splitting."""
        print("\n" + "="*60)
        print("LOADING DATASETS")
        print("="*60)
        
        train_df = pd.read_csv(Config.TRAIN_CSV).reset_index(drop=True)
        val_df = pd.read_csv(Config.VAL_CSV).reset_index(drop=True)
        test_df = pd.read_csv(Config.TEST_CSV).reset_index(drop=True)

#----------------------------------------------------------------------------REMOVE SAMPLING - USE FULL DATASET
        # train_df = train_df.sample(frac=0.0001, random_state=42) 
        # val_df = val_df.sample(frac=0.0001, random_state=42)      
        # test_df = test_df.sample(frac=0.0001, random_state=42)  # USE FULL TEST SET for reliable KPIs
#---------------------------------------------------------------------------------------------------------------         
        
        # Ensure proper label encoding (0=benign, 1=malicious)
        for df in [train_df, val_df, test_df]:
            if 'label' in df.columns:
                if df['label'].dtype == 'object':
                    df['label'] = df['label'].map({
                        'legit': 0, 'benign': 0, 'legitimate': 0,
                        'malicious': 1, 'phishing': 1, 'phish': 1
                    })
                df['label'] = df['label'].fillna(0).astype(int)
        
        print(f"\nDataset Sizes:")
        print(f"  Train:      {len(train_df):,}")
        print(f"  Validation: {len(val_df):,}")
        print(f"  Test:       {len(test_df):,}")
        print(f"  Total:      {len(train_df) + len(val_df) + len(test_df):,}")
        
        return train_df, val_df, test_df
        
    def load_datasets(self) -> Tuple[URLDataset, URLDataset, URLDataset]:
        train_df, val_df, test_df = self.load_dataframes()
        train_dataset = URLDataset(train_df, self.tokenizer)
        val_dataset = URLDataset(val_df, self.tokenizer)
        test_dataset = URLDataset(test_df, self.tokenizer)
        return train_dataset, val_dataset, test_dataset
    
    
    def create_model(self) -> nn.Module:
        """Build MiniLM-L12-H384 model with LoRA."""
        print("\n" + "="*60)
        print("BUILDING MODEL")
        print("="*60)
        
        base_model = MiniLMURLClassifier()
        apply_structured_pruning(base_model, Config.PRUNING_RATIO)
        
        lora_config = LoraConfig(
            task_type="SEQ_CLS",
            inference_mode=False,
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            lora_dropout=Config.LORA_DROPOUT,
            target_modules=Config.LORA_TARGET_MODULES
        )
        
        model = get_peft_model(base_model, lora_config)
        model = model.to(Config.DEVICE)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel Architecture: MiniLM-L12-H384 + LoRA")
        print(f"  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"  Frozen parameters:    {total_params - trainable_params:,}")
        
        return model
    
    def train(self) -> bool:
        """Execute complete training pipeline with Stratified K-Fold and checkpoint resuming."""
        print("\n" + "="*80)
        print("MiniLM PHISHING DETECTION TRAINING PIPELINE")
        print("="*80)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {Config.DEVICE}")
        print(f"Target: <{Config.MAX_MODEL_SIZE_MB}MB model with 98% accuracy")
        print("="*80)
        
        train_df, val_df, test_df = self.load_dataframes()
        test_dataset = URLDataset(test_df, self.tokenizer)
        test_loader = DataLoader( test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=Config.PIN_MEMORY, prefetch_factor=2 if 2 > 0 else None)
        
        from sklearn.model_selection import StratifiedKFold
        if Config.USE_STRATIFIED_KFOLD:
            print("\n" + "="*60)
            print(f"USING STRATIFIED {Config.KFOLD_SPLITS}-FOLD CROSS VALIDATION")
            print("="*60)
            combined_train_val_df = pd.concat([train_df, val_df], ignore_index=True)
            skf = StratifiedKFold(n_splits=Config.KFOLD_SPLITS, shuffle=True, random_state=Config.SEED)
            folds = list(skf.split(np.zeros(len(combined_train_val_df)), combined_train_val_df['label']))
        else:
            combined_train_val_df = pd.concat([train_df, val_df], ignore_index=True)
            folds = [(np.arange(len(train_df)), np.arange(len(train_df), len(combined_train_val_df)))]
            
        overall_best_kpi = 0.0
        overall_best_fold = 0
        overall_kpi_compliance = False
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            if Config.USE_STRATIFIED_KFOLD:
                print(f"\n" + "="*80)
                print(f"[START] STARTING FOLD {fold_idx + 1}/{Config.KFOLD_SPLITS}")
                print(f"="*80)
                
            fold_save_root = Config.SAVE_ROOT / f"fold_{fold_idx+1}" if Config.USE_STRATIFIED_KFOLD else Config.SAVE_ROOT
            fold_save_root.mkdir(parents=True, exist_ok=True)
            self.checkpoint_manager = CheckpointManager(fold_save_root / "checkpoints")
            
            fold_train_df = combined_train_val_df.iloc[train_idx].reset_index(drop=True)
            fold_val_df = combined_train_val_df.iloc[val_idx].reset_index(drop=True)
            
            train_dataset = URLDataset(fold_train_df, self.tokenizer)
            val_dataset = URLDataset(fold_val_df, self.tokenizer)
            
            model = self.create_model()
            
            if Config.USE_WEIGHTED_SAMPLING:
                sampler = create_weighted_sampler(train_dataset.labels)
                train_loader = DataLoader( train_dataset, batch_size=Config.BATCH_SIZE, sampler=sampler, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY, prefetch_factor=Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None)
                print(f"[OK] Training with weighted sampling (balanced batches)")
            else:
                train_loader = DataLoader( train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY, prefetch_factor=Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None)
                print(f"[OK] Training with standard random shuffling")
            
            val_loader = DataLoader( val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=Config.PIN_MEMORY, prefetch_factor=2 if 2 > 0 else None)
            
            optimizer = optim.AdamW( model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY, eps=1e-8)
            total_steps = len(train_loader) * Config.NUM_EPOCHS
            warmup_steps = int(Config.LR_WARMUP_RATIO * total_steps)
            scheduler = get_cosine_schedule_with_warmup( optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps, num_cycles=0.5, last_epoch=-1)
            criterion = FocalLoss().to(Config.DEVICE)
            scaler = GradScaler(enabled=Config.USE_AMP)
            
            start_epoch = 1
            best_kpi_score = 0.0
            best_model_epoch = 0
            patience_counter = 0
            
            latest_checkpoint = self.checkpoint_manager.find_latest_checkpoint()
            
            if latest_checkpoint:
                print(f"\n[RESUME] CHECKPOINT FOUND for Fold {fold_idx+1}: {latest_checkpoint.name}")
                start_epoch, last_metrics, best_kpi_score, loaded_history = \
                    self.checkpoint_manager.load_checkpoint(
                        latest_checkpoint, model, optimizer, scheduler, scaler
                    )
                if loaded_history:
                    self.training_history = loaded_history
                
                best_model_dirs = sorted(fold_save_root.glob("best_model_epoch_*"))
                if best_model_dirs:
                    best_model_epoch = int(best_model_dirs[-1].name.split("_")[-1])
                else:
                    best_model_epoch = start_epoch - 1
            else:
                self.training_history = {
                    'train_losses': [], 'val_losses': [],
                    'train_accs': [], 'val_accs': [],
                    'kpi_scores': [], 'thresholds': []
                }
            
            epoch = start_epoch - 1
            for epoch in range(start_epoch, Config.NUM_EPOCHS + 1):
                print(f"\n{'='*60}")
                print(f"FOLD {fold_idx+1} - EPOCH {epoch}/{Config.NUM_EPOCHS}")
                print(f"{'='*60}")
                
                train_loss, train_acc = self._train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, epoch)
                val_loss, val_probs, val_labels = self._validate_epoch(model, val_loader, criterion)
                
                optimal_threshold, _threshold_info = self.kpi_evaluator.find_optimal_threshold_strict(val_labels, val_probs)
                val_preds = (val_probs >= optimal_threshold).astype(int)
                val_metrics = self.kpi_evaluator.evaluate_metrics(val_labels, val_preds, val_probs)
                
                self._update_history(train_loss, train_acc, val_loss, val_metrics, optimal_threshold)
                self._print_epoch_summary(epoch, train_loss, train_acc, val_loss, val_metrics, optimal_threshold)
                
                if val_metrics['kpi_score'] > best_kpi_score:
                    best_kpi_score = val_metrics['kpi_score']
                    best_model_epoch = epoch
                    patience_counter = 0
                    self._save_best_model(model, epoch, val_metrics, optimal_threshold, fold_save_root)
                    print(f"[BEST] New best model for Fold {fold_idx+1}! KPI Score improved to {best_kpi_score:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= Config.PATIENCE:
                        print(f"\n⏸ Early stopping triggered at epoch {epoch}")
                        break
                
                self.checkpoint_manager.save_checkpoint( model, optimizer, scheduler, scaler, epoch, val_metrics, optimal_threshold, best_kpi_score, self.training_history)
                self.checkpoint_manager.cleanup_old_checkpoints(keep_last_n=3)
            
            if best_kpi_score > overall_best_kpi:
                overall_best_kpi = best_kpi_score
                overall_best_fold = fold_idx + 1
                
            print("\n" + "="*80)
            print(f"FINAL TEST EVALUATION FOR FOLD {fold_idx+1}")
            print("="*80)
            fold_kpi_compliance = self._evaluate_test_set(model, test_loader, criterion, best_model_epoch, fold_save_root)
            if fold_kpi_compliance and overall_best_fold == fold_idx + 1:
                overall_kpi_compliance = True
                
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Overall Best Model: Fold {overall_best_fold} (KPI Score: {overall_best_kpi:.4f})")
        print(f"KPI Compliance: {'[PASS] ACHIEVED' if overall_kpi_compliance else '[WARN] PARTIAL'}")
        print("="*80 + "\n")
        
        return overall_kpi_compliance
        
    def _train_epoch( self, model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, scheduler: Any, criterion: nn.Module, scaler: GradScaler, epoch: int) -> Tuple[float, float]:
        """Train for one epoch with proper gradient accumulation."""
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Training")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels = batch['labels'].to(Config.DEVICE)
            
            # Forward pass
            with autocast(enabled=Config.USE_AMP):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                loss = loss / Config.GRAD_ACCUM_STEPS  # Scale loss
            
            # Backward pass
            if Config.USE_AMP:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every GRAD_ACCUM_STEPS
            if (batch_idx + 1) % Config.GRAD_ACCUM_STEPS == 0:
                if Config.USE_AMP:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
            
            # Metrics
            running_loss += loss.item() * labels.size(0) * Config.GRAD_ACCUM_STEPS
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f"{loss.item() * Config.GRAD_ACCUM_STEPS:.4f}",
                'acc': f"{accuracy_score(all_labels[-len(preds):], preds):.4f}"
            })
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def _validate_epoch( self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, np.ndarray, np.ndarray]:
        """Validation with NaN checks."""
        model.eval()
        running_loss = 0.0
        all_probs, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(Config.DEVICE)
                attention_mask = batch['attention_mask'].to(Config.DEVICE)
                labels = batch['labels'].to(Config.DEVICE)
                
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print("[WARN] NaN/Inf loss detected, skipping batch")
                    continue
                
                running_loss += loss.item() * labels.size(0)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                
                all_probs.extend(probs[:, 1])
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader.dataset) if len(all_probs) > 0 else float('inf')
        return epoch_loss, np.array(all_probs), np.array(all_labels)
    
    def _update_history( self, train_loss: float, train_acc: float, val_loss: float, val_metrics: Dict, threshold: float) -> None:
        """Update training history."""
        self.training_history['train_losses'].append(train_loss)
        self.training_history['val_losses'].append(val_loss)
        self.training_history['train_accs'].append(train_acc)
        self.training_history['val_accs'].append(val_metrics['accuracy'])
        self.training_history['kpi_scores'].append(val_metrics['kpi_score'])
        self.training_history['thresholds'].append(threshold)
    
    def _print_epoch_summary( self, epoch: int, train_loss: float, train_acc: float, val_loss: float, val_metrics: Dict, threshold: float) -> None:
        """Print comprehensive epoch results."""
        print(f"\nResults:")
        print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Optimal Threshold: {threshold:.4f}")
        
        print(f"\nKPI Metrics:")
        kpi_checks = {
            'Accuracy':  (val_metrics['accuracy'],  Config.TARGET_ACCURACY,  '>='),
            'Precision': (val_metrics['precision'], Config.TARGET_PRECISION, '>='),
            'Recall':    (val_metrics['recall'],    Config.TARGET_RECALL,    '>='),
            'FNR':       (val_metrics['fnr'],       Config.MAX_FNR,          '<='),
            'FPR':       (val_metrics['fpr'],       Config.MAX_FPR,          '<=')
        }
        
        for name, (value, target, op) in kpi_checks.items():
            passed = (value >= target) if op == '>=' else (value <= target)
            symbol = '[PASS]' if passed else '[FAIL]'
            print(f"  {name:<12} {value:.4f} (target: {op}{target:.4f}) {symbol}")
        
        status = "[PASS] ALL KPIs MET" if val_metrics['kpi_compliance'] else "[WARN] KPIs NOT MET"
        print(f"\nStatus: {status} (Score: {val_metrics['kpi_score']:.4f})")
    
    def _save_best_model( self, model: nn.Module, epoch: int, metrics: Dict, threshold: float, save_root: Path = Config.SAVE_ROOT) -> None:
        """Save best model with all exports and artifacts."""
        best_model_dir = save_root / f"best_model_epoch_{epoch:03d}"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"[SAVE] SAVING BEST MODEL - EPOCH {epoch}")
        print(f"{'='*60}")
        
        # Save LoRA adapter and tokenizer
        model.save_pretrained(best_model_dir / "lora_adapter")
        self.tokenizer.save_pretrained(best_model_dir)
        print(f"[OK] LoRA adapter saved")
        print(f"[OK] Tokenizer saved")
        
        # Save full model
        full_model_path = best_model_dir / "model_full.pt"
        torch.save(model, full_model_path)
        model_size = os.path.getsize(full_model_path) / (1024 * 1024)
        print(f"[OK] Full model saved: {model_size:.2f} MB")
        
        # Save state dict
        # torch.save(model.state_dict(), best_model_dir / "model_state_dict.pt")
        # print(f"[OK] State dict saved")
        
        # Merge and export
        print(f"\nExporting production models...")
        merged_model, merged_size = ModelExporter.merge_lora_and_export(model, self.tokenizer, best_model_dir)
        
        final_size = ModelExporter.export_onnx(merged_model, best_model_dir)
        
        if final_size and final_size <= Config.MAX_MODEL_SIZE_MB:
            print(f"\n[SUCCESS] SUCCESS: Model size {final_size:.2f}MB meets {Config.MAX_MODEL_SIZE_MB}MB target!")
        
        # Save training artifacts
        artifact_saver = ArtifactSaver(best_model_dir)
        artifact_saver.save_history(
            self.training_history['train_losses'],
            self.training_history['val_losses'],
            self.training_history['train_accs'],
            self.training_history['val_accs']
        )
        
        # Save metadata
        metadata = {
            'model_info': {
                'epoch': epoch,
                'architecture': 'MiniLM v3 Base + LoRA',
                'base_model': Config.MODEL_NAME,
                'max_length': Config.MAX_LEN,
                'model_size_mb': model_size,
                'quantized_size_mb': final_size
            },
            'performance': metrics,
            'threshold': threshold,
            'training_config': CheckpointManager._serialize_config(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(best_model_dir / "deployment_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"\n[PASS] Best model saved to: {best_model_dir.name}")
        print(f"{'='*60}\n")
    

    def inference_from_checkpoint(self) -> bool:
        """
        Inference-only mode: Load latest checkpoint and perform test evaluation.
        Skips all training.
        """
        print("\n" + "="*80)
        print("MINILM PHISHING DETECTION - INFERENCE MODE (CHECKPOINT RESUME)")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {Config.DEVICE}")
        print("="*80 + "\n")
        
        # ========================================
        # LOAD DATASETS
        # ========================================
        print("Loading datasets...")
        test_dataset = self.load_datasets()[2]  # Only need test dataset
        test_loader = DataLoader(
            test_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=Config.PIN_MEMORY, 
            prefetch_factor=2 if 2 > 0 else None
        )
        print(f"[OK] Test dataset loaded: {len(test_dataset):,} samples\n")
        
        # ========================================
        # FIND AND LOAD LATEST CHECKPOINT
        # ========================================
        print("="*60)
        print("CHECKPOINT SEARCH")
        print("="*60)
        
        latest_checkpoint = self.checkpoint_manager.find_latest_checkpoint()
        
        if latest_checkpoint is None:
            print("[FAIL] ERROR: No checkpoint found!")
            print(f"Expected checkpoint directory: {Config.CHECKPOINT_DIR}")
            print("Please run training first before attempting inference.\n")
            return False
        
        print(f"[PASS] Found checkpoint: {latest_checkpoint.name}")
        
        # ========================================
        # LOAD MODEL AND CHECKPOINT STATE
        # ========================================
        print("\nLoading model and checkpoint state...")
        model = self.create_model()
        criterion = FocalLoss().to(Config.DEVICE)
        
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=Config.DEVICE, weights_only=False)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.to(Config.DEVICE)
            print(f"[OK] Model state loaded")
            
            # Restore training history
            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']
                print(f"[OK] Training history restored ({len(self.training_history['train_losses'])} epochs)")
            
            # Extract metadata
            checkpoint_epoch = checkpoint.get('epoch', 0)
            best_kpi_score = checkpoint.get('best_kpi_score', 0.0)
            
            print(f"[OK] Checkpoint epoch: {checkpoint_epoch}")
            print(f"[OK] Best KPI score at checkpoint: {best_kpi_score:.4f}\n")
            
        except Exception as e:
            print(f"[FAIL] Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # ========================================
        # FIND BEST MODEL EPOCH
        # ========================================
        # The checkpoint epoch may not be the best model epoch
        # Scan for best_model_epoch_* directories to find the actual best model
        best_model_dirs = sorted(Config.SAVE_ROOT.glob("best_model_epoch_*"))
        if best_model_dirs:
            # Get the latest best model directory (highest epoch number)
            best_model_dir = best_model_dirs[-1]
            # Extract epoch number from directory name (e.g., "best_model_epoch_001" -> 1)
            best_model_epoch = int(best_model_dir.name.split("_")[-1])
            print(f"[OK] Best model found at epoch: {best_model_epoch}")
        else:
            # Fallback to checkpoint epoch if no best model directory found
            best_model_epoch = checkpoint_epoch
            print(f"[WARN] No best_model_epoch_* directory found, using checkpoint epoch: {best_model_epoch}")
        
        # ========================================
        # RUN TEST INFERENCE
        # ========================================
        print("="*60)
        print("TEST INFERENCE")
        print("="*60 + "\n")
        
        model.eval()
        kpi_compliance = self._evaluate_test_set(model, test_loader, criterion, best_model_epoch)
        
        # ========================================
        # SUMMARY
        # ========================================
        print("\n" + "="*80)
        print("INFERENCE COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Checkpoint Used: {latest_checkpoint.name}")
        print(f"Checkpoint Epoch: {checkpoint_epoch}")
        print(f"KPI Compliance: {'[PASS] ACHIEVED' if kpi_compliance else '[WARN] PARTIAL'}")
        print(f"Results Directory: {Config.SAVE_ROOT / 'final_test_evaluation'}")
        print("="*80 + "\n")
        
        return kpi_compliance

    def _evaluate_test_set(self, model: nn.Module, test_loader: DataLoader, criterion: nn.Module, best_epoch: int, save_root: Path = Config.SAVE_ROOT) -> bool:
        """Final evaluation on test set using production-ready merged model."""
        test_inference_dir = save_root / "final_test_evaluation"
        test_inference_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("LOADING PRODUCTION MODEL FOR TEST INFERENCE")
        print(f"{'='*60}")
        
        # ========================================
        # LOAD MERGED PRODUCTION MODEL
        # ========================================
        best_model_dir = save_root / f"best_model_epoch_{best_epoch:03d}"
        merged_model_path = best_model_dir / "model_merged_full.pt"
        
        if merged_model_path.exists():
            try:
                print(f"Loading merged model: {merged_model_path.name}")
                model = torch.load(merged_model_path, map_location=Config.DEVICE, weights_only=False)
                model.eval()
                print(f"[OK] Using production-ready merged model (LoRA weights integrated)")
            except Exception as e:
                print(f"[WARN] Failed to load merged model: {e}")
                print(f"[WARN] Falling back to training model")
                model.eval()
        else:
            print(f"[WARN] Merged model not found at {merged_model_path}")
            print(f"[WARN] Using training model (with LoRA)")
            model.eval()
        
        print(f"Results will be saved to: {test_inference_dir.name}")
        print(f"{'='*60}\n")
        
        # ========================================
        # RUN TEST INFERENCE
        # ========================================
        all_probs, all_labels, all_urls = [], [], []
        test_running_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test Inference"):
                input_ids = batch['input_ids'].to(Config.DEVICE)
                attention_mask = batch['attention_mask'].to(Config.DEVICE)
                labels = batch['labels'].to(Config.DEVICE)
                
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                
                test_running_loss += loss.item() * labels.size(0)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                
                all_probs.extend(probs[:, 1])
                all_labels.extend(labels.cpu().numpy())
                all_urls.extend(batch['url'])
        
        # ========================================
        # COMPUTE METRICS
        # ========================================
        test_loss = test_running_loss / len(test_loader.dataset)
        test_probs = np.array(all_probs)
        test_labels = np.array(all_labels)
        
        optimal_threshold = self.training_history['thresholds'][-1] if self.training_history['thresholds'] else 0.5
        test_preds = (test_probs >= optimal_threshold).astype(int)
        
        test_metrics = self.kpi_evaluator.evaluate_metrics(test_labels, test_preds, test_probs)
        test_metrics['test_loss'] = test_loss
        test_metrics['threshold_used'] = optimal_threshold
        test_metrics['model_used'] = 'model_merged_full.pt' if merged_model_path.exists() else 'model_full.pt (with LoRA)'
        
        # ========================================
        # SAVE RESULTS
        # ========================================
        # Save predictions
        predictions_df = pd.DataFrame({
            'url': all_urls,
            'true_label': test_labels,
            'predicted_label': test_preds,
            'prob_malicious': test_probs,
            'correct': test_labels == test_preds
        })
        predictions_df.to_csv(test_inference_dir / "test_predictions.csv", index=False)
        print(f"[OK] Predictions saved: test_predictions.csv")
        
        # Save artifacts
        artifact_saver = ArtifactSaver(test_inference_dir)
        artifact_saver.save_test_metrics(test_metrics, optimal_threshold)
        artifact_saver.save_test_plots(test_labels, test_probs, optimal_threshold)
        
        # ========================================
        # PRINT SUMMARY
        # ========================================
        print(f"\n{'='*60}")
        print("TEST SET RESULTS")
        print(f"{'='*60}")
        print(f"Model Used: {test_metrics['model_used']}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Threshold: {optimal_threshold:.4f}")
        print(f"\nMetrics:")
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f} {'[PASS]' if test_metrics['accuracy'] >= Config.TARGET_ACCURACY else '[FAIL]'}")
        print(f"  Precision: {test_metrics['precision']:.4f} {'[PASS]' if test_metrics['precision'] >= Config.TARGET_PRECISION else '[FAIL]'}")
        print(f"  Recall:    {test_metrics['recall']:.4f} {'[PASS]' if test_metrics['recall'] >= Config.TARGET_RECALL else '[FAIL]'}")
        print(f"  F1-Score:  {test_metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {test_metrics['auc']:.4f}")
        print(f"\nError Rates:")
        print(f"  FNR: {test_metrics['fnr']:.4f} {'[PASS]' if test_metrics['fnr'] <= Config.MAX_FNR else '[FAIL]'}")
        print(f"  FPR: {test_metrics['fpr']:.4f} {'[PASS]' if test_metrics['fpr'] <= Config.MAX_FPR else '[FAIL]'}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {test_metrics['tn']:,}  |  FP: {test_metrics['fp']:,}")
        print(f"  FN: {test_metrics['fn']:,}  |  TP: {test_metrics['tp']:,}")
        print(f"\nKPI Compliance: {'[PASS] ACHIEVED' if test_metrics['kpi_compliance'] else '[FAIL] NOT MET'}")
        print(f"{'='*60}")
        
        # ========================================
        # SAVE FINAL RESULTS
        # ========================================
        hyperparams = CheckpointManager._serialize_config()
        
        final_results = {
            'test_metrics': test_metrics,
            'training_history': self.training_history,
            'best_epoch': best_epoch,
            'optimal_threshold': optimal_threshold,
            'kpi_compliance': test_metrics['kpi_compliance'],
            'model_architecture': 'MiniLM v3 Base',
            'model_used_for_test': test_metrics['model_used'],
            'test_samples': len(test_labels),
            'timestamp': datetime.now().isoformat(),
            'hyperparameters': hyperparams
        }
        
        with open(save_root / "final_results.json", 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\n[OK] All test results saved to: {test_inference_dir.name}")
        print(f"[OK] Final results: final_results.json")
        
        return test_metrics['kpi_compliance']


    def onnx_inference(self, onnx_model_type: str = 'int8') -> bool:
        """
        ONNX Inference Mode: Load ONNX model and evaluate on test set.
        
        This mode uses the production-ready ONNX model for inference.
        No PyTorch model is loaded; inference runs entirely through ONNX Runtime.
        
        Args:
            onnx_model_type: 'int8' for quantized, 'fp32' for original, or custom path
        """
        print("\n" + "="*80)
        print("MINILM PHISHING DETECTION - ONNX INFERENCE MODE (INT8 QUANTIZED)")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Runtime: ONNX Runtime (CPU-optimized)")
        print("="*80 + "\n")
        
        # ========================================
        # VALIDATE ONNX RUNTIME AVAILABILITY
        # ========================================
        try:
            import onnxruntime as ort
            print(f"[OK] ONNX Runtime version: {ort.__version__}")
            available_providers = ort.get_available_providers()
            print(f"[OK] Available providers: {available_providers}")
        except ImportError:
            print("[FAIL] ERROR: onnxruntime not installed!")
            print("Install with: pip install onnxruntime")
            print("For GPU support: pip install onnxruntime-gpu")
            return False
        
        # ========================================
        # FIND ONNX MODEL
        # ========================================
        print(f"\n{'='*60}")
        print("ONNX MODEL SEARCH")
        print(f"{'='*60}")
        
        # Search for best_model_epoch_* directories
        best_model_dirs = sorted(Config.SAVE_ROOT.glob("best_model_epoch_*"))
        
        if not best_model_dirs:
            print("[FAIL] ERROR: No best_model_epoch_* directory found!")
            print(f"Expected in: {Config.SAVE_ROOT}")
            print("Please run training first (python MiniLM_1.py --mode train)")
            return False
        
        best_model_dir = best_model_dirs[-1]
        best_epoch = int(best_model_dir.name.split("_")[-1])
        print(f"[OK] Best model directory: {best_model_dir.name}")
        print(f"[OK] Best model epoch: {best_epoch}")
        
        # Look for ONNX model based on user selection
        onnx_quant_path = best_model_dir / "model_quant_8bit.onnx"
        onnx_fp32_path = best_model_dir / "model.onnx"
        
        # Determine which model to use
        if os.path.isfile(onnx_model_type):
            # Custom path provided
            onnx_model_path = Path(onnx_model_type)
            model_type = f"Custom ({onnx_model_path.name})"
            model_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
            print(f"[PASS] Using custom ONNX model: {onnx_model_path}")
            print(f"   Model size: {model_size:.2f} MB")
        elif onnx_model_type == 'fp32':
            if onnx_fp32_path.exists():
                onnx_model_path = onnx_fp32_path
                model_type = "FP32 (Full Precision)"
                model_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
                print(f"[PASS] Using FP32 ONNX model: {onnx_fp32_path.name}")
                print(f"   Model size: {model_size:.2f} MB")
            else:
                print(f"[FAIL] ERROR: FP32 ONNX model not found: {onnx_fp32_path}")
                return False
        else:  # 'int8' (default)
            if onnx_quant_path.exists():
                onnx_model_path = onnx_quant_path
                model_type = "INT8 Quantized"
                model_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
                print(f"[PASS] Found quantized ONNX model: {onnx_quant_path.name}")
                print(f"   Model size: {model_size:.2f} MB")
                if model_size <= Config.MAX_MODEL_SIZE_MB:
                    print(f"   [PASS] Meets {Config.MAX_MODEL_SIZE_MB} MB deployment target!")
                else:
                    print(f"   [WARN] Exceeds {Config.MAX_MODEL_SIZE_MB} MB target ({model_size:.2f} MB)")
            elif onnx_fp32_path.exists():
                onnx_model_path = onnx_fp32_path
                model_type = "FP32 (INT8 not available)"
                model_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
                print(f"[WARN] INT8 model not found, falling back to FP32: {onnx_fp32_path.name}")
                print(f"   Model size: {model_size:.2f} MB")
            else:
                print("[FAIL] ERROR: No ONNX model found!")
                print(f"   Searched: {onnx_quant_path}")
                print(f"   Searched: {onnx_fp32_path}")
                print("Please ensure ONNX export was successful during training.")
                return False
        
        print(f"{'='*60}\n")
        
        # ========================================
        # LOAD DATASETS
        # ========================================
        print("Loading test dataset...")
        test_dataset = self.load_datasets()[2]  # Only need test dataset
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=False,  # CPU inference for ONNX
            prefetch_factor=2 if 2 > 0 else None
        )
        print(f"[OK] Test dataset loaded: {len(test_dataset):,} samples\n")
        
        # ========================================
        # LOAD DEPLOYMENT METADATA (for threshold)
        # ========================================
        optimal_threshold = None  # Will be set from metadata
        
        # Try to load threshold from deployment metadata
        metadata_path = best_model_dir / "deployment_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                # Check both possible key names
                for key in ['threshold', 'optimal_threshold']:
                    if key in metadata and metadata[key] is not None:
                        optimal_threshold = float(metadata[key])
                        print(f"[OK] Loaded threshold from deployment_metadata.json['{key}']: {optimal_threshold:.4f}")
                        break
                # Also check nested under performance
                if optimal_threshold is None and 'performance' in metadata:
                    perf = metadata['performance']
                    for key in ['threshold', 'optimal_threshold']:
                        if key in perf and perf[key] is not None:
                            optimal_threshold = float(perf[key])
                            print(f"[OK] Loaded threshold from metadata.performance['{key}']: {optimal_threshold:.4f}")
                            break
            except Exception as e:
                print(f"[WARN] Could not load metadata: {e}")
        
        # Also try from final_results.json
        if optimal_threshold is None:
            results_path = Config.SAVE_ROOT / "final_results.json"
            if results_path.exists():
                try:
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                    for key in ['optimal_threshold', 'threshold']:
                        if key in results and results[key] is not None:
                            optimal_threshold = float(results[key])
                            print(f"[OK] Loaded threshold from final_results.json['{key}']: {optimal_threshold:.4f}")
                            break
                except Exception as e:
                    print(f"[WARN] Could not load results: {e}")
        
        # Also try from training history in checkpoint
        if optimal_threshold is None:
            latest_checkpoint = self.checkpoint_manager.find_latest_checkpoint()
            if latest_checkpoint is not None:
                try:
                    ckpt = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
                    if 'training_history' in ckpt:
                        thresholds = ckpt['training_history'].get('thresholds', [])
                        if thresholds:
                            optimal_threshold = float(thresholds[-1])
                            print(f"[OK] Loaded threshold from checkpoint training history: {optimal_threshold:.4f}")
                except Exception as e:
                    print(f"[WARN] Could not load checkpoint for threshold: {e}")
        
        # Final fallback
        if optimal_threshold is None:
            optimal_threshold = 0.5
            print(f"[WARN] No threshold found in any source, using default: {optimal_threshold:.4f}")
        
        # ========================================
        # CREATE ONNX SESSION
        # ========================================
        print(f"\n{'='*60}")
        print("ONNX SESSION INITIALIZATION")
        print(f"{'='*60}")
        
        # Configure session options for maximum performance
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = Config.NUM_WORKERS
        sess_options.inter_op_num_threads = 2
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # ── Pre-load cuDNN/CUDA libraries from pip packages ─────
        # When cuDNN is installed via pip (nvidia-cudnn-cu12), the .so files
        # live inside site-packages/nvidia/cudnn/lib/ and are NOT on
        # LD_LIBRARY_PATH by default. We must preload them with ctypes
        # so ONNX Runtime's dlopen() can find them.
        # Reference: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#preload-dlls
        cuda_usable = False
        if 'CUDAExecutionProvider' in available_providers:
            # Method 1: ort.preload_dlls() (onnxruntime >= 1.21.0)
            try:
                if hasattr(ort, 'preload_dlls'):
                    ort.preload_dlls(cuda=True, cudnn=True)
                    print("[OK] Preloaded CUDA/cuDNN via ort.preload_dlls()")
                    cuda_usable = True
            except Exception as e:
                print(f"[WARN] ort.preload_dlls() failed: {e}")
            
            # Method 2: Manually locate nvidia pip package lib dirs and preload .so files
            if not cuda_usable:
                import ctypes
                nvidia_lib_dirs = []
                
                # --- Find nvidia/cudnn/lib via multiple strategies ---
                # Strategy A: Use importlib to find the nvidia.cudnn package spec
                try:
                    import importlib.util
                    spec = importlib.util.find_spec('nvidia.cudnn')
                    if spec and spec.submodule_search_locations:
                        for loc in spec.submodule_search_locations:
                            candidate = Path(loc) / 'lib'
                            if candidate.exists():
                                nvidia_lib_dirs.append(str(candidate))
                except Exception:
                    pass
                
                # Strategy B: Walk site-packages looking for nvidia/cudnn/lib
                if not nvidia_lib_dirs:
                    try:
                        import site
                        site_dirs = site.getsitepackages() + [site.getusersitepackages()]
                        for sp in site_dirs:
                            candidate = Path(sp) / 'nvidia' / 'cudnn' / 'lib'
                            if candidate.exists():
                                nvidia_lib_dirs.append(str(candidate))
                            # Also check nvidia/cuda_runtime/lib
                            cuda_rt = Path(sp) / 'nvidia' / 'cuda_runtime' / 'lib'
                            if cuda_rt.exists():
                                nvidia_lib_dirs.append(str(cuda_rt))
                    except Exception:
                        pass
                
                # Strategy C: Derive from sys.prefix (conda environments)
                if not nvidia_lib_dirs:
                    conda_sp = Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
                    for sub in ['nvidia/cudnn/lib', 'nvidia/cuda_runtime/lib']:
                        candidate = conda_sp / sub
                        if candidate.exists():
                            nvidia_lib_dirs.append(str(candidate))
                
                # Also add torch's bundled libraries
                try:
                    torch_lib_dir = Path(torch.__file__).parent / 'lib'
                    if torch_lib_dir.exists():
                        nvidia_lib_dirs.append(str(torch_lib_dir))
                except Exception:
                    pass
                
                # Deduplicate while preserving order
                seen = set()
                nvidia_lib_dirs = [d for d in nvidia_lib_dirs if d not in seen and not seen.add(d)]
                
                if nvidia_lib_dirs:
                    print(f"[OK] Found {len(nvidia_lib_dirs)} NVIDIA library dirs:")
                    for d in nvidia_lib_dirs:
                        print(f"  → {d}")
                    
                    # Pre-load ALL cuDNN .so files with RTLD_GLOBAL so dlopen() finds them
                    loaded_count = 0
                    cudnn_libs = [
                        'libcudnn.so.9', 'libcudnn_adv.so.9', 'libcudnn_ops.so.9',
                        'libcudnn_cnn.so.9', 'libcudnn_graph.so.9',
                        'libcudnn_engines_precompiled.so.9',
                        'libcudnn_engines_runtime_compiled.so.9',
                        'libcudnn_heuristic.so.9'
                    ]
                    for lib_dir in nvidia_lib_dirs:
                        for lib_name in cudnn_libs:
                            lib_path = os.path.join(lib_dir, lib_name)
                            if os.path.exists(lib_path):
                                try:
                                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                                    loaded_count += 1
                                except OSError as e:
                                    print(f"  [WARN] Failed to load {lib_name}: {e}")
                    
                    print(f"[OK] Pre-loaded {loaded_count} cuDNN libraries into process memory")
                    
                    if loaded_count > 0:
                        cuda_usable = True
                    else:
                        print("[WARN] No cuDNN .so files could be loaded - falling back to CPU")
                else:
                    print("[WARN] CUDAExecutionProvider listed but cuDNN 9.x libraries not found")
                    print("  → Falling back to CPUExecutionProvider")
                    print("  → To enable GPU: pip install nvidia-cudnn-cu12")
        
        if cuda_usable:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        # Suppress ONNX Runtime internal warnings during session creation
        ort.set_default_logger_severity(3)  # 3 = ERROR only, suppresses WARN
        
        try:
            session = ort.InferenceSession(
                str(onnx_model_path),
                sess_options=sess_options,
                providers=providers
            )
            
            # Restore normal logging after session creation
            ort.set_default_logger_severity(1)  # 1 = default (VERBOSE)
            
            # Detect ACTUAL active provider
            active_provider = session.get_providers()[0]
            if active_provider == 'CUDAExecutionProvider':
                execution_device = "CUDA (GPU)"
            elif active_provider == 'TensorrtExecutionProvider':
                execution_device = "TensorRT (GPU)"
            else:
                execution_device = "CPU"
            
            print(f"[OK] ONNX session created successfully")
            print(f"  Active provider: {active_provider}")
            print(f"  Execution device: {execution_device}")
            print(f"  Graph optimization: ENABLED (all levels)")
            print(f"  Intra-op threads: {Config.NUM_WORKERS}")
            
            # Print input/output info
            input_names = [inp.name for inp in session.get_inputs()]
            output_names = [out.name for out in session.get_outputs()]
            print(f"  Input names: {input_names}")
            print(f"  Output names: {output_names}")
        except Exception as e:
            ort.set_default_logger_severity(1)  # Restore logging on failure too
            print(f"[FAIL] Failed to create ONNX session: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"{'='*60}\n")
        
        # ========================================
        # RUN ONNX INFERENCE
        # ========================================
        print(f"{'='*60}")
        print("ONNX TEST INFERENCE")
        print(f"{'='*60}\n")
        
        all_probs, all_labels, all_urls = [], [], []
        inference_times = []
        
        import time
        total_start = time.perf_counter()
        
        for batch in tqdm(test_loader, desc="ONNX Inference"):
            input_ids = batch['input_ids'].numpy()
            attention_mask = batch['attention_mask'].numpy()
            labels = batch['labels'].numpy()
            
            # Build ONNX input feed
            ort_inputs = {
                input_names[0]: input_ids,
                input_names[1]: attention_mask
            }
            
            # Run inference and measure time
            batch_start = time.perf_counter()
            ort_outputs = session.run(output_names, ort_inputs)
            batch_end = time.perf_counter()
            
            inference_times.append(batch_end - batch_start)
            
            # Extract logits and compute probabilities
            logits = ort_outputs[0]
            
            # Softmax to get probabilities
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            all_probs.extend(probs[:, 1])
            all_labels.extend(labels)
            all_urls.extend(batch['url'])
        
        total_end = time.perf_counter()
        total_inference_time = total_end - total_start
        
        # ========================================
        # COMPUTE METRICS
        # ========================================
        test_probs = np.array(all_probs)
        test_labels = np.array(all_labels)
        test_preds = (test_probs >= optimal_threshold).astype(int)
        
        test_metrics = self.kpi_evaluator.evaluate_metrics(test_labels, test_preds, test_probs)
        test_metrics['model_used'] = f'ONNX {model_type} ({onnx_model_path.name})'
        test_metrics['threshold_used'] = optimal_threshold
        
        # Compute detailed timing metrics
        total_samples = len(test_labels)
        avg_batch_time_ms = np.mean(inference_times) * 1000
        avg_sample_time_ms = (total_inference_time / total_samples) * 1000
        throughput = total_samples / total_inference_time
        
        # ========================================
        # SAVE RESULTS
        # ========================================
        onnx_results_dir = Config.SAVE_ROOT / "onnx_test_evaluation"
        onnx_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'url': all_urls,
            'true_label': test_labels,
            'predicted_label': test_preds,
            'prob_malicious': test_probs,
            'correct': test_labels == test_preds
        })
        predictions_df.to_csv(onnx_results_dir / "onnx_test_predictions.csv", index=False)
        print(f"\n[OK] ONNX predictions saved: onnx_test_predictions.csv")
        
        # Save artifacts (plots and metrics)
        artifact_saver = ArtifactSaver(onnx_results_dir)
        artifact_saver.save_test_metrics(test_metrics, optimal_threshold)
        artifact_saver.save_test_plots(test_labels, test_probs, optimal_threshold)
        
        # ========================================
        # PRINT COMPREHENSIVE RESULTS
        # ========================================
        print(f"\n{'='*80}")
        print("ONNX INFERENCE RESULTS")
        print(f"{'='*80}")
        print(f"Model: {onnx_model_path.name} ({model_type})")
        print(f"Model Size: {model_size:.2f} MB {'[PASS]' if model_size <= Config.MAX_MODEL_SIZE_MB else '[FAIL]'}")
        print(f"Best Epoch: {best_epoch}")
        print(f"Threshold: {optimal_threshold:.4f}")
        
        print(f"\n{'─'*40}")
        print("CLASSIFICATION METRICS")
        print(f"{'─'*40}")
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f} {'[PASS]' if test_metrics['accuracy'] >= Config.TARGET_ACCURACY else '[FAIL]'}")
        print(f"  Precision: {test_metrics['precision']:.4f} {'[PASS]' if test_metrics['precision'] >= Config.TARGET_PRECISION else '[FAIL]'}")
        print(f"  Recall:    {test_metrics['recall']:.4f} {'[PASS]' if test_metrics['recall'] >= Config.TARGET_RECALL else '[FAIL]'}")
        print(f"  F1-Score:  {test_metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {test_metrics['auc']:.4f}")
        
        print(f"\n{'─'*40}")
        print("ERROR RATES")
        print(f"{'─'*40}")
        print(f"  FNR: {test_metrics['fnr']:.4f} {'[PASS]' if test_metrics['fnr'] <= Config.MAX_FNR else '[FAIL]'}")
        print(f"  FPR: {test_metrics['fpr']:.4f} {'[PASS]' if test_metrics['fpr'] <= Config.MAX_FPR else '[FAIL]'}")
        
        print(f"\n{'─'*40}")
        print("CONFUSION MATRIX")
        print(f"{'─'*40}")
        print(f"  TN: {test_metrics['tn']:,}  |  FP: {test_metrics['fp']:,}")
        print(f"  FN: {test_metrics['fn']:,}  |  TP: {test_metrics['tp']:,}")
        
        print(f"\n{'─'*40}")
        print("PERFORMANCE BENCHMARKS")
        print(f"{'─'*40}")
        print(f"  Total inference time:    {total_inference_time:.2f}s")
        print(f"  Total samples:           {total_samples:,}")
        print(f"  Avg batch time:          {avg_batch_time_ms:.2f} ms")
        print(f"  Avg per-sample time:     {avg_sample_time_ms:.3f} ms")
        print(f"  Throughput:              {throughput:.0f} URLs/sec")
        print(f"  Execution device:        {execution_device}")
        
        print(f"\n{'─'*40}")
        print("KPI COMPLIANCE")
        print(f"{'─'*40}")
        kpi_pass = test_metrics['kpi_compliance']
        size_pass = model_size <= Config.MAX_MODEL_SIZE_MB
        full_compliance = kpi_pass and size_pass
        
        print(f"  Accuracy ≥ {Config.TARGET_ACCURACY:.0%}:   {'[PASS] PASS' if test_metrics['accuracy'] >= Config.TARGET_ACCURACY else '[FAIL] FAIL'}")
        print(f"  Precision ≥ {Config.TARGET_PRECISION:.0%}:  {'[PASS] PASS' if test_metrics['precision'] >= Config.TARGET_PRECISION else '[FAIL] FAIL'}")
        print(f"  Recall ≥ {Config.TARGET_RECALL:.0%}:     {'[PASS] PASS' if test_metrics['recall'] >= Config.TARGET_RECALL else '[FAIL] FAIL'}")
        print(f"  FPR ≤ {Config.MAX_FPR:.0%}:         {'[PASS] PASS' if test_metrics['fpr'] <= Config.MAX_FPR else '[FAIL] FAIL'}")
        print(f"  FNR ≤ {Config.MAX_FNR:.0%}:        {'[PASS] PASS' if test_metrics['fnr'] <= Config.MAX_FNR else '[FAIL] FAIL'}")
        print(f"  Size ≤ {Config.MAX_MODEL_SIZE_MB}MB:     {'[PASS] PASS' if size_pass else '[FAIL] FAIL'} ({model_size:.2f} MB)")
        print(f"\n  Overall: {'[PASS] ALL KPIs MET - PRODUCTION READY' if full_compliance else '[FAIL] KPIs NOT FULLY MET'}")
        print(f"{'='*80}")
        
        # ========================================
        # COMPARE WITH PYTORCH RESULTS (if available)
        # ========================================
        pytorch_results_path = Config.SAVE_ROOT / "final_results.json"
        if pytorch_results_path.exists():
            try:
                with open(pytorch_results_path, 'r') as f:
                    pytorch_results = json.load(f)
                
                pt_metrics = pytorch_results.get('test_metrics', {})
                
                if pt_metrics:
                    print(f"\n{'='*80}")
                    print("ONNX INT8 vs PyTorch MODEL COMPARISON")
                    print(f"{'='*80}")
                    print(f"{'Metric':<20} {'PyTorch':>12} {'ONNX INT8':>12} {'Δ Delta':>12}")
                    print(f"{'─'*56}")
                    
                    comparison_metrics = [
                        ('Accuracy', 'accuracy'),
                        ('Precision', 'precision'),
                        ('Recall', 'recall'),
                        ('F1-Score', 'f1'),
                        ('AUC-ROC', 'auc'),
                        ('FPR', 'fpr'),
                        ('FNR', 'fnr'),
                    ]
                    
                    for name, key in comparison_metrics:
                        pt_val = pt_metrics.get(key, 0)
                        onnx_val = test_metrics.get(key, 0)
                        delta = onnx_val - pt_val
                        
                        # Use appropriate delta indicator
                        if key in ['fpr', 'fnr']:
                            # Lower is better for error rates
                            delta_icon = "🟢" if delta <= 0 else "🔴"
                        else:
                            # Higher is better for performance metrics
                            delta_icon = "🟢" if delta >= 0 else "🔴"
                        
                        print(f"  {name:<18} {pt_val:>11.4f} {onnx_val:>11.4f} {delta:>+11.4f} {delta_icon}")
                    
                    print(f"{'─'*56}")
                    print(f"  {'Model Size':<18} {'N/A':>12} {model_size:>10.2f}MB")
                    print(f"  {'Throughput':<18} {'N/A':>12} {throughput:>8.0f}/sec")
                    print(f"{'='*80}")
                    
            except Exception as e:
                print(f"\n[WARN] Could not load PyTorch results for comparison: {e}")
        
        # ========================================
        # SAVE ONNX INFERENCE RESULTS
        # ========================================
        onnx_final_results = {
            'test_metrics': test_metrics,
            'model_info': {
                'model_path': str(onnx_model_path),
                'model_type': model_type,
                'model_size_mb': model_size,
                'best_epoch': best_epoch,
                'onnx_runtime_version': ort.__version__,
                'execution_provider': session.get_providers()[0],
            },
            'performance_benchmarks': {
                'total_inference_time_sec': total_inference_time,
                'total_samples': total_samples,
                'avg_batch_time_ms': avg_batch_time_ms,
                'avg_sample_time_ms': avg_sample_time_ms,
                'throughput_urls_per_sec': throughput,
                'execution_device': execution_device,
            },
            'kpi_compliance': {
                'classification_kpis_met': kpi_pass,
                'size_kpi_met': size_pass,
                'all_kpis_met': full_compliance,
            },
            'threshold': optimal_threshold,
            'timestamp': datetime.now().isoformat(),
        }
        
        onnx_results_file = Config.SAVE_ROOT / "onnx_inference_results.json"
        with open(onnx_results_file, 'w') as f:
            json.dump(onnx_final_results, f, indent=2, default=str)
        
        print(f"\n[OK] All ONNX results saved to: {onnx_results_dir.name}/")
        print(f"[OK] ONNX results JSON: {onnx_results_file.name}")
        
        # ========================================
        # FINAL SUMMARY
        # ========================================
        print(f"\n{'='*80}")
        print("ONNX INFERENCE COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"Model: {onnx_model_path.name} ({model_type}, {model_size:.2f} MB)")
        print(f"Throughput: {throughput:.0f} URLs/sec")
        print(f"KPI Compliance: {'[PASS] ALL MET - PRODUCTION READY' if full_compliance else '[WARN] PARTIAL'}")
        print(f"Results: {onnx_results_dir}")
        print(f"{'='*80}\n")
        
        return full_compliance


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def setup_cli_parser() -> argparse.ArgumentParser:
    """Setup CLI argument parser with mode selection."""
    parser = argparse.ArgumentParser(
        description="MiniLM Phishing URL Detection - Training & Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
USAGE EXAMPLES:
  
  1. Train from scratch (or resume from latest checkpoint):
     python MiniLM_1.py --mode train
     
  2. Train and allow checkpoint resume prompt:
     python MiniLM_1.py --mode train --interactive
     
  3. Inference-only mode (skip training, load latest checkpoint):
     python MiniLM_1.py --mode inference
     
  4. ONNX inference (quantized INT8 model - production deployment test):
     python MiniLM_1.py --mode onnx-inference
     
  5. Default mode (auto-detect based on checkpoints):
     python MiniLM_1.py
     
  6. Show help:
     python MiniLM_1.py --help
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'inference', 'onnx-inference', 'auto'],
        default='auto',
        help="""
        Execution mode:
        - 'train': Full training pipeline (default behavior)
        - 'inference': Load latest checkpoint and perform test inference only
        - 'onnx-inference': Load quantized ONNX model and evaluate on test set
        - 'auto': Detect mode based on checkpoint existence (default)
        """
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        default=False,
        help="Enable interactive prompts (e.g., checkpoint resume confirmation)"
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help="Path to specific checkpoint to load (optional)"
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help="Override number of training epochs (if None, uses Config.NUM_EPOCHS)"
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help="Override batch size (if None, uses Config.BATCH_SIZE)"
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help="Override learning rate (if None, uses Config.LR)"
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help="Enable verbose output with additional logging"
    )
    
    parser.add_argument(
        '--onnx-model',
        type=str,
        default='int8',
        help="""
        ONNX model variant for onnx-inference mode:
        - 'int8': INT8 quantized model (model_quant_8bit.onnx) - smallest, may have accuracy loss
        - 'fp32': FP32 original ONNX model (model.onnx) - same accuracy as PyTorch
        - '/path/to/model.onnx': Custom ONNX model path
        (default: int8)
        """
    )
    
    return parser


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main() -> bool:
    """Main entry point with CLI support."""
    
    # Parse CLI arguments
    parser = setup_cli_parser()
    args = parser.parse_args()
    
    # Override Config if CLI arguments provided
    if args.epochs is not None:
        Config.NUM_EPOCHS = args.epochs
        print(f"[CLI] Overriding NUM_EPOCHS to {args.epochs}")
    
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
        print(f"[CLI] Overriding BATCH_SIZE to {args.batch_size}")
    
    if args.lr is not None:
        Config.LR = args.lr
        print(f"[CLI] Overriding LR to {args.lr}")
    
    # Determine execution mode
    mode = args.mode
    
    if mode == 'auto':
        # Auto-detect mode based on checkpoint existence
        Config.setup_paths()
        latest_ckpt = CheckpointManager(Config.CHECKPOINT_DIR).find_latest_checkpoint()
        
        if latest_ckpt is not None:
            # Checkpoint exists - default to inference mode
            print("\n[AUTO-DETECT] Latest checkpoint found")
            print(f"[AUTO-DETECT] Setting mode to 'inference'\n")
            mode = 'inference'
        else:
            # No checkpoint - default to training mode
            print("\n[AUTO-DETECT] No checkpoint found")
            print(f"[AUTO-DETECT] Setting mode to 'train'\n")
            mode = 'train'
    
    # ========================================
    # TRAINING MODE
    # ========================================
    if mode == 'train':
        try:
            print("\n" + "="*80)
            print(" " * 20 + "MiniLM PHISHING URL DETECTION")
            print(" " * 15 + "Production-Grade Training Pipeline")
            print("="*80)
            print(f"Mode:                  TRAINING")
            print(f"Target Model Size:     <{Config.MAX_MODEL_SIZE_MB}MB with {Config.TARGET_ACCURACY:.0%} accuracy")
            print(f"Architecture:          MiniLM v3 Base + LoRA + Focal Loss")
            print(f"Device:                {Config.DEVICE}")
            print(f"Interactive Mode:      {'Enabled' if args.interactive else 'Disabled'}")
            print("="*80 + "\n")
            
            trainer = PhishingDetectionTrainer()
            kpi_compliance = trainer.train()
            
            return kpi_compliance
        
        except KeyboardInterrupt:
            print("\n\n[WARN] Training interrupted by user")
            return False
        
        except Exception as e:
            print(f"\n[FAIL] ERROR during training: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========================================
    # INFERENCE MODE (PyTorch)
    # ========================================
    elif mode == 'inference':
        try:
            trainer = PhishingDetectionTrainer()
            kpi_compliance = trainer.inference_from_checkpoint()
            
            return kpi_compliance
        
        except KeyboardInterrupt:
            print("\n\n[WARN] Inference interrupted by user")
            return False
        
        except Exception as e:
            print(f"\n[FAIL] ERROR during inference: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========================================
    # ONNX INFERENCE MODE (INT8 Quantized)
    # ========================================
    elif mode == 'onnx-inference':
        try:
            print("\n" + "="*80)
            print(" " * 15 + "MiniLM PHISHING URL DETECTION")
            print(" " * 10 + "ONNX Runtime Inference (INT8 Quantized)")
            print("="*80)
            print(f"Mode:                  ONNX INFERENCE")
            print(f"Target Model Size:     <{Config.MAX_MODEL_SIZE_MB}MB")
            print(f"Runtime:               ONNX Runtime")
            print(f"Quantization:          INT8 Dynamic")
            print("="*80 + "\n")
            
            trainer = PhishingDetectionTrainer()
            kpi_compliance = trainer.onnx_inference(onnx_model_type=args.onnx_model)
            
            return kpi_compliance
        
        except KeyboardInterrupt:
            print("\n\n[WARN] ONNX inference interrupted by user")
            return False
        
        except Exception as e:
            print(f"\n[FAIL] ERROR during ONNX inference: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    else:
        print(f"[FAIL] Unknown mode: {mode}")
        parser.print_help()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)




# python 2_MiniLM_V4_Model_On_Raw_data_and_OFP_and_Canonical_inferencing_v4.py --mode train
# python 2_MiniLM_V4_Model_On_Raw_data_and_OFP_and_Canonical_inferencing_v4.py --mode inference


#################### python 2_MiniLM_V4_Model_On_Raw_data_and_OFP_and_Canonical_inferencing_v4.py --mode onnx-inference
# # FP32 ONNX (same accuracy as PyTorch, larger file)
# python 2_MiniLM_V4_Model_On_Raw_data_and_OFP_and_Canonical_inferencing_v4.py --mode onnx-inference --onnx-model fp32

# # INT8 Quantized (smaller file, may lose accuracy)
# python 2_MiniLM_V4_Model_On_Raw_data_and_OFP_and_Canonical_inferencing_v4.py --mode onnx-inference --onnx-model int8

# # Custom model path
# python 2_MiniLM_V4_Model_On_Raw_data_and_OFP_and_Canonical_inferencing_v4.py --mode onnx-inference --onnx-model /path/to/model.onnx