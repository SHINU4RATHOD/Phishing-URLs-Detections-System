import os
import json
import sys
from pathlib import Path
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
from core.model import MiniLMURLClassifier, save_model_summary

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
            mlflow_run_id = checkpoint.get('mlflow_run_id', None)
            
            print(f"[OK] Model state restored")
            print(f"[OK] Resuming from epoch {start_epoch}")
            print(f"[OK] Best KPI score: {best_kpi_score:.4f}")
            print(f"{'='*60}\n")
            
            return start_epoch, metrics, best_kpi_score, training_history, mlflow_run_id
        
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
    
    def save_test_plots(self, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> None:
        """Generate test set visualizations."""
        y_pred = (y_prob >= threshold).astype(int)
        
        self._plot_confusion_matrix(y_true, y_pred, threshold)
        self._plot_roc_curve(y_true, y_prob)
        self._plot_pr_curve(y_true, y_prob)
        print(f"[OK] Test plots saved")
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> None:
        """Plot confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        cm_percent = cm / total * 100
        
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_percent, annot=False, fmt='.1f', cmap='Blues', xticklabels=['Benign', 'Malicious'], yticklabels=['Benign', 'Malicious'], cbar_kws={'label': 'Percentage'})
        
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
            adapter_path = save_dir / "lora_adapter"
            base_model = MiniLMURLClassifier(vocab_size=len(tokenizer))
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
                dynamo=False,
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
            print(f"[FAIL] ONNX export failed: {e}")
            return None
    
    @staticmethod
    def _quantize_onnx(onnx_path: Path, save_dir: Path) -> Optional[float]:
        """Quantize ONNX model to 8-bit."""
        try:
            print("Quantizing ONNX model...")
            quant_path = save_dir / "model_quant_8bit.onnx"
            
            # Use in-memory shape inference to avoid Windows file locks on model-inferred.onnx
            import onnx
            from onnx.shape_inference import infer_shapes
            
            onnx_model = onnx.load(str(onnx_path))
            inferred_model = infer_shapes(onnx_model)
            
            quantize_dynamic(
                inferred_model,
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
