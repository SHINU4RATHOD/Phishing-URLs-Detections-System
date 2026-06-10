import os
import json
import re
import sys
import time
import ipaddress
from pathlib import Path
from collections import Counter
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from peft import PeftModel

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve
)
from core.config import Config
from core.model import HybridGLUClassifier, save_model_summary

# ONNX Quantization Setup
ONNX_QUANTIZATION_AVAILABLE = False
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_QUANTIZATION_AVAILABLE = True
except ImportError:
    pass


class CheckpointManager:
    """Handles model checkpointing with resume support."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, scheduler: Optional[Any], 
                        scaler: Optional[GradScaler], epoch: int, metrics: Dict, threshold: float, 
                        best_kpi_score: float, training_history: Dict, mlflow_run_id: Optional[str] = None) -> Path:
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
            'config': config_dict,
            'mlflow_run_id': mlflow_run_id
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        temp_path = checkpoint_path.with_suffix('.pt.tmp')
        
        try:
            torch.save(checkpoint, temp_path)
            temp_path.rename(checkpoint_path)
            print(f"[OK] Checkpoint saved: {checkpoint_path.name}")
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Path, model: nn.Module, optimizer: Optional[optim.Optimizer] = None, 
                        scheduler: Optional[Any] = None, scaler: Optional[GradScaler] = None) -> Tuple[int, Dict, float, Dict, Optional[str]]:
        """Load checkpoint and restore full training state."""
        try:
            print(f"\n{'='*60}")
            print(f"RESUMING FROM CHECKPOINT")
            print(f"{'='*60}")
            
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
            mlflow_run_id = checkpoint.get('mlflow_run_id', None)
            
            print(f"[OK] Model state restored")
            print(f"[OK] Resuming from epoch {start_epoch}")
            print(f"[OK] Best KPI score: {best_kpi_score:.4f}")
            print(f"{'='*60}\n")
            
            return start_epoch, metrics, best_kpi_score, training_history, mlflow_run_id
        
        except Exception as e:
            print(f"[WARN] Checkpoint load failed ({checkpoint_path.name}): {e}")
            print(f"[WARN] Keeping the checkpoint file for inspection. Starting fresh this run.")
            return 1, {}, 0.0, {}, None
    
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
                    print(f"[CLEANUP] Cleaned up old checkpoint: {ckpt.name}")
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


class ArtifactSaver:
    """Saves training artifacts, plots, and metrics."""
    
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    def save_history(self, train_losses: List[float], val_losses: List[float], 
                     train_accs: List[float], val_accs: List[float]) -> None:
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
    
    def save_test_plots(self, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> None:
        """Generate test set visualizations."""
        y_pred = (y_prob >= threshold).astype(int)
        
        self._plot_confusion_matrix(y_true, y_pred, threshold)
        self._plot_roc_curve(y_true, y_prob)
        self._plot_pr_curve(y_true, y_prob)
        print(f"[OK] Test plots saved")
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> None:
        """Plot confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        cm_percent = cm / total * 100
        
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_percent, annot=False, fmt='.1f', cmap='Blues', xticklabels=['Benign', 'Malicious'], yticklabels=['Benign', 'Malicious'], cbar_kws={'label': 'Percentage'})
        
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
        plt.title(f'Test Confusion Matrix (Threshold={threshold:.3f})\n' f'FNR={fnr:.2%} | FPR={fpr:.2%}', fontsize=16, fontweight='bold', pad=20)
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


class ModelExporter:
    """Handles model export to ONNX with quantization."""
    
    @staticmethod
    def merge_lora_and_export(model: nn.Module, tokenizer, save_dir: Path) -> Tuple[nn.Module, float]:
        """Merge LoRA adapters and save production model."""
        try:
            print("Merging LoRA adapters...")
            
            full_model_path = save_dir / "model_full.pt"
            loaded_model = torch.load(full_model_path, map_location=Config.DEVICE, weights_only=False)
            
            merged_model = loaded_model.merge_and_unload()
            merged_model = merged_model.to(Config.DEVICE).eval()
            
            merged_path = save_dir / "model_merged_full.pt"
            torch.save(merged_model, merged_path)
            merged_size = os.path.getsize(merged_path) / (1024 * 1024)
            
            print(f"[OK] Merged model: {merged_size:.2f} MB")
            
            torch.save(merged_model.state_dict(), save_dir / "model_merged_state_dict.pt")
            
            summary_path = save_dir / "model_summery.txt"
            save_model_summary(merged_model, input_size=(1, Config.MAX_LEN), save_path=str(summary_path))
            
            return merged_model, merged_size
        
        except Exception as e:
            print(f"[WARN] LoRA merge failed: {e}")
            return model, 0.0
    
    @staticmethod
    def export_onnx(model: nn.Module, save_dir: Path) -> Optional[float]:
        """Export model to ONNX format with quantization."""
        if not Config.EXPORT_ONNX:
            return None
        
        try:
            print("Exporting to ONNX...")
            model.eval()
            device = next(model.parameters()).device
            
            vocab_size = model.encoder.config.vocab_size
            dummy_input = {
                'input_ids': torch.randint(0, vocab_size, (1, Config.MAX_LEN), dtype=torch.long).to(device),
                'attention_mask': torch.ones(1, Config.MAX_LEN, dtype=torch.long).to(device),
                'heuristic_features': torch.randn(1, Config.HEURISTIC_DIM, dtype=torch.float32).to(device)
            }
            
            onnx_path = save_dir / "model.onnx"
            torch.onnx.export(
                model,
                (dummy_input['input_ids'], dummy_input['attention_mask'], dummy_input['heuristic_features']),
                str(onnx_path),
                opset_version=Config.ONNX_OPSET,
                input_names=['input_ids', 'attention_mask', 'heuristic_features'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch', 1: 'sequence'},
                    'attention_mask': {0: 'batch', 1: 'sequence'},
                    'heuristic_features': {0: 'batch'},
                    'logits': {0: 'batch'}
                },
                dynamo=False,
                verbose=False
            )
            
            onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"[OK] ONNX model: {onnx_size:.2f} MB")
            
            if Config.EXPORT_QUANTIZED and ONNX_QUANTIZATION_AVAILABLE:
                quant_size = ModelExporter._quantize_onnx(onnx_path, save_dir)
                return quant_size if quant_size else onnx_size
            
            return onnx_size
        
        except Exception as e:
            print(f"[FAIL] ONNX export failed: {e}")
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
            return quant_size
        
        except Exception as e:
            print(f"[WARN] ONNX quantization failed: {e}")
            return None


# ============================================================================
# UNSUPPORTED URL SCREENING LAYER
# ============================================================================
INFERENCE_FILTER_SCHEME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
INFERENCE_FILTER_WINDOWS_DRIVE_PATTERN = re.compile(r"^[a-zA-Z]:[\\/]")
INFERENCE_FILTER_DECIMAL_IP_PATTERN = re.compile(r"^\d{8,10}$")
INFERENCE_FILTER_HEX_IP_PATTERN = re.compile(r"^(?:0x)?[0-9A-Fa-f]{8}$")

INFERENCE_URL_SHORTENERS = {
    "bit.ly", "bitly.com", "goo.gl", "t.co", "tinyurl.com", "ow.ly", "is.gd", "v.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "tiny.cc", "t.ly", "rb.gy", "shrtco.de",
    "s.id", "clck.ru", "bl.ink", "shorturl.at", "adf.ly", "q.gs", "short.io",
    "short.cm", "soo.gd", "lnkd.in", "x.co", "mcaf.ee", "amzn.to", "trib.al",
    "smarturl.it", "snip.ly", "snipurl.com", "lnk.bio", "lnk.to", "bio.link",
    "bio.site", "tap.bio", "linktr.ee", "beacons.ai", "campsite.bio", "hey.bio",
    "instabio.cc", "yt.be", "youtu.be", "link.medium.com", "reut.rs", "nyti.ms",
    "shorte.st", "clk.sh", "clk.im", "linkshrink.net", "bc.vc", "adcrun.ch",
    "ouo.io", "exe.io", "exey.io", "linkvertise.com", "shrinkme.io",
    "shrinkearn.com", "shortzon.com", "cutpaid.com", "cutwin.com", "shortadd.com",
    "short.pe", "shrinkurl.io", "urlcash.net", "t.me", "telegram.me", "wa.me",
    "goo.su", "urlzs.com", "linkvertise.net", "go.microsoft.com", "redirect.vk.com",
    "vk.cc", "vk.me", "msft.it", "msft.ms", "aka.ms", "shortzy.in", "shortxlink.com",
    "ez4short.com", "gtly.to", "sharee.tech", "stfly.io", "zws.im", "dlink.me",
    "hyperurl.co", "urlr.me", "clicky.me", "linkbox.to", "tiny.lt", "myurls.co",
    "shortbitly.com", "shortbit.com", "urlbitly.net", "bit.do", "short.best",
    "clickmeter.com", "urlr.in", "go.ly", "click.ru", "shortenurl.io", "taplink.at",
    "goo.by", "shortly.cc", "shortcm.li", "urlz.fr", "sk.gy"
}

INFERENCE_SUSPICIOUS_PORTS = {
    8080, 8081, 8082, 8083, 8084, 8085, 8088, 8000, 8001, 8008, 8010, 8443, 8880,
    8888, 9999, 1080, 3128, 9050, 8118, 1081, 1085, 4145, 4153, 6588, 6589, 6666,
    6667, 6697, 22, 2222, 23, 3389, 5900, 5985, 5986, 3306, 5432, 27017, 6379,
    9200, 11211, 1337, 1338, 1352, 4443, 4444, 5555, 6660, 7000, 7001, 8089,
    9000, 9001, 9002, 9003, 9010, 10000, 10101, 10443, 12345, 16000, 22222, 25,
    465, 587, 2525, 110, 995, 143, 993, 2082, 2083, 2095, 2096, 8890, 9090, 9443,
    9998, 10080
}

INFERENCE_URL_CATEGORY_SPECS = [
    {
        "name": "Shortened_URL",
        "policy": "filtered",
        "filter_during_inference": True,
        "description": "Short-form redirector URLs cannot be expanded offline during inference.",
    },
    {
        "name": "IP-Based URL",
        "policy": "filtered",
        "filter_during_inference": True,
        "description": "Direct-IP and suspicious port targets.",
    },
    {
        "name": "Chrome_Internal_URL",
        "policy": "filtered",
        "filter_during_inference": True,
        "description": "Browser-internal chrome:// URLs.",
    },
    {
        "name": "Data_URL",
        "policy": "filtered",
        "filter_during_inference": True,
        "description": "Embedded base64 data payloads.",
    },
    {
        "name": "FTP_SFTP_URL",
        "policy": "filtered",
        "filter_during_inference": True,
        "description": "ftp:// and sftp:// protocol schemes.",
    },
    {
        "name": "File_URL",
        "policy": "filtered",
        "filter_during_inference": True,
        "description": "Local file paths and UNC drive scopes.",
    },
    {
        "name": "JavaScript_URL",
        "policy": "filtered",
        "filter_during_inference": True,
        "description": "javascript: pseudo-executable payloads.",
    },
    {
        "name": "Telegram_Bot_URL",
        "policy": "filtered",
        "filter_during_inference": True,
        "description": "Telegram bot redirection APIs.",
    },
]
INFERENCE_URL_CATEGORY_SPEC_MAP = {spec["name"]: spec for spec in INFERENCE_URL_CATEGORY_SPECS}


def _normalize_inference_screening_url(raw_url: Any) -> str:
    """Normalize raw strings into a urlparse-compatible form for offline screening."""
    candidate = str(raw_url or "").strip()
    if not candidate:
        return "http://"
    if INFERENCE_FILTER_SCHEME_PATTERN.match(candidate):
        return candidate
    lower_candidate = candidate.lower()
    if lower_candidate.startswith((
        "javascript:", "data:", "file:", "blob:", "chrome:", "about:", "edge:",
        "ftp:", "sftp:"
    )):
        return candidate
    if candidate.startswith("\\\\") or INFERENCE_FILTER_WINDOWS_DRIVE_PATTERN.match(candidate.replace("/", "\\")):
        return "file:///" + candidate.lstrip('/\\')
    return f"http://{candidate}"


def _extract_inference_screening_parts(raw_url: Any) -> Tuple[str, Any, str, str, str, Optional[int]]:
    from urllib.parse import urlparse
    normalized = _normalize_inference_screening_url(raw_url)
    parsed = urlparse(normalized)
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower().strip(".")
    host = host[4:] if host.startswith("www.") else host
    path = parsed.path or ""
    try:
        port = parsed.port
    except ValueError:
        port = None
    return normalized, parsed, scheme, host, path, port


def detect_inference_screening_categories(raw_url: Any) -> List[str]:
    """Detect reportable categories for test-set screening before inference."""
    normalized, parsed, scheme, host, path, port = _extract_inference_screening_parts(raw_url)
    raw_text = str(raw_url or "")
    path_lower = path.lower()
    query_lower = (parsed.query or "").lower()

    categories: List[str] = []

    if host in INFERENCE_URL_SHORTENERS:
        categories.append("Shortened_URL")

    ip_based = False
    if host:
        try:
            ipaddress.ip_address(host)
            ip_based = True
        except ValueError:
            host_no_dot = host.replace(".", "")
            try:
                if INFERENCE_FILTER_DECIMAL_IP_PATTERN.fullmatch(host_no_dot):
                    ipaddress.IPv4Address(int(host_no_dot))
                    ip_based = True
                elif INFERENCE_FILTER_HEX_IP_PATTERN.fullmatch(host_no_dot):
                    ipaddress.IPv4Address(int(host_no_dot, 16))
                    ip_based = True
            except Exception:
                pass
    if port in INFERENCE_SUSPICIOUS_PORTS:
        ip_based = True
    if ip_based:
        categories.append("IP-Based URL")

    if scheme == "chrome" or normalized.lower().startswith("chrome://"):
        categories.append("Chrome_Internal_URL")
    if scheme == "data":
        categories.append("Data_URL")
    if scheme in {"ftp", "sftp"}:
        categories.append("FTP_SFTP_URL")
    if scheme == "file" or raw_text.startswith("\\\\") or INFERENCE_FILTER_WINDOWS_DRIVE_PATTERN.match(raw_text.replace("/", "\\")):
        categories.append("File_URL")
    if scheme == "javascript":
        categories.append("JavaScript_URL")
    if host == "api.telegram.org" and "/bot" in path_lower:
        categories.append("Telegram_Bot_URL")
    elif host in {"t.me", "telegram.me"}:
        if "bot" in path_lower:
            categories.append("Telegram_Bot_URL")
        elif any(token in path_lower for token in ("start=", "startgroup=", "startapp=")):
            categories.append("Telegram_Bot_URL")
        elif any(token in query_lower for token in ("start=", "startgroup=", "startapp=")):
            categories.append("Telegram_Bot_URL")

    return categories


def filter_inference_unsupported_test_urls(test_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Filter unsupported URL categories from the test split and build a structured report."""
    category_counts = Counter({spec["name"]: 0 for spec in INFERENCE_URL_CATEGORY_SPECS})
    matched_any_mask: List[bool] = []
    filtered_mask: List[bool] = []

    # Safe lookup for input column (supports 'input' and 'url')
    url_col = 'input' if 'input' in test_df.columns else 'url'

    for raw_url in test_df[url_col].fillna("").astype(str).tolist():
        matches = detect_inference_screening_categories(raw_url)
        matched_any_mask.append(bool(matches))
        should_filter = False
        for category in matches:
            category_counts[category] += 1
            spec = INFERENCE_URL_CATEGORY_SPEC_MAP[category]
            if spec["filter_during_inference"]:
                should_filter = True
        filtered_mask.append(should_filter)

    filtered_mask_np = np.array(filtered_mask, dtype=bool)
    filtered_df = test_df.loc[~filtered_mask_np].reset_index(drop=True)

    report_categories = []
    for spec in INFERENCE_URL_CATEGORY_SPECS:
        matched = int(category_counts[spec["name"]])
        filtered = matched if spec["filter_during_inference"] else 0
        retained = matched - filtered
        report_categories.append({
            "name": spec["name"],
            "policy": spec["policy"],
            "matched_count": matched,
            "filtered_count": filtered,
            "retained_count": retained,
            "description": spec["description"],
        })

    report = {
        "report_name": "unsupported_url_category_report",
        "generated_at": datetime.now().isoformat(),
        "total_urls_scanned": int(len(test_df)),
        "urls_matching_reported_categories": int(sum(matched_any_mask)),
        "total_urls_filtered": int(filtered_mask_np.sum()),
        "total_urls_retained": int(len(filtered_df)),
        "category_counting_note": "Category counts are non-exclusive; a single URL can contribute to multiple buckets.",
        "categories": report_categories,
    }
    return filtered_df, report


def format_inference_url_screening_report(report: Optional[Dict[str, Any]]) -> str:
    """Render a developer-oriented console report for unsupported test URLs."""
    if not report:
        return ""

    lines = [
        "",
        "=" * 80,
        "UNSUPPORTED URL CATEGORY REPORT (TEST INFERENCE)",
        "=" * 80,
        f"Dataset scanned:                    {report['total_urls_scanned']:,}",
        f"URLs matched by reported buckets:   {report['urls_matching_reported_categories']:,}",
        f"URLs filtered before inference:     {report['total_urls_filtered']:,}",
        f"URLs retained for inference:        {report['total_urls_retained']:,}",
        report["category_counting_note"],
        "",
        f"{'Category':<24} {'Policy':<24} {'Matched':>12} {'Filtered':>12} {'Retained':>12}",
        "-" * 88,
    ]

    for entry in report["categories"]:
        lines.append(
            f"{entry['name']:<24} {entry['policy']:<24} "
            f"{entry['matched_count']:>12,} {entry['filtered_count']:>12,} {entry['retained_count']:>12,}"
        )
        lines.append(f"  Detail: {entry['description']}")

    lines.append("=" * 80)
    return "\n".join(lines)


def save_inference_url_screening_report(report: Optional[Dict[str, Any]], output_dir: Path) -> None:
    """Persist the screening report next to inference artifacts."""
    if not report:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    report_json_path = output_dir / "unsupported_url_category_report.json"
    report_txt_path = output_dir / "unsupported_url_category_report.txt"

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(format_inference_url_screening_report(report) + "\n")
