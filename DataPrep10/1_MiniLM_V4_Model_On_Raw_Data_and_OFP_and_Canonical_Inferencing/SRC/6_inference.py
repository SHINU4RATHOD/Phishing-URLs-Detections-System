import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm

# Ensure sibling packages under SRC can be imported regardless of run directory
sys.path.append(str(Path(__file__).resolve().parent))

from core.config import Config
from core.dataset import URLDataset
from core.loss import FocalLoss
from core.model import MiniLMURLClassifier
from core.evaluator import EnhancedKPIEvaluator
from core.utils import CheckpointManager, ArtifactSaver


class PhishingDetectionInference:
    """Main inference orchestrator supporting PyTorch checkpoints and ONNX Runtime."""
    
    def __init__(self):
        Config.setup_reproducibility()
        Config.setup_paths()
        
        if Config.USE_CUSTOM_TOKENIZER_BYTE_LEVEL_BPE or Config.USE_CUSTOM_TOKENIZER_SENTENCEPIECE_UNIGRAM_BPE:
            from transformers import PreTrainedTokenizerFast
            tokenizer_path = Path(Config.CUSTOM_TOKENIZER_PATH)
            if not tokenizer_path.exists():
                raise FileNotFoundError(f"Custom tokenizer not found at {tokenizer_path}. Please train the model first.")
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(tokenizer_path),
                unk_token="[UNK]",
                cls_token="[CLS]",
                sep_token="[SEP]",
                pad_token="[PAD]",
                mask_token="[MASK]"
            )
            print(f"[OK] Loaded custom tokenizer from {tokenizer_path}, vocab size: {len(self.tokenizer)}")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        
        self.checkpoint_manager = CheckpointManager(Config.CHECKPOINT_DIR)
        self.kpi_evaluator = EnhancedKPIEvaluator()
        
        self.training_history = {
            'train_losses': [], 'val_losses': [],
            'train_accs': [], 'val_accs': [],
            'kpi_scores': [], 'thresholds': []
        }

    def load_datasets(self) -> Tuple[URLDataset, URLDataset, URLDataset]:
        """Load datasets for final test evaluation."""
        print("Loading datasets...")
        train_df = pd.read_csv(Config.TRAIN_CSV).reset_index(drop=True)
        val_df = pd.read_csv(Config.VAL_CSV).reset_index(drop=True)
        test_df = pd.read_csv(Config.TEST_CSV).reset_index(drop=True)
        
        # Ensure proper label encoding (0=benign, 1=malicious)
        for df in [train_df, val_df, test_df]:
            if 'label' in df.columns:
                if df['label'].dtype == 'object':
                    df['label'] = df['label'].map({
                        'legit': 0, 'benign': 0, 'legitimate': 0,
                        'malicious': 1, 'phishing': 1, 'phish': 1
                    })
                df['label'] = df['label'].fillna(0).astype(int)
                
        train_dataset = URLDataset(train_df, self.tokenizer)
        val_dataset = URLDataset(val_df, self.tokenizer)
        test_dataset = URLDataset(test_df, self.tokenizer)
        return train_dataset, val_dataset, test_dataset

    def create_model(self, vocab_size: Optional[int] = None) -> nn.Module:
        """Build base MiniLM-L12-H384 model."""
        base_model = MiniLMURLClassifier(vocab_size=vocab_size)
        return base_model

    def inference_from_checkpoint(self) -> bool:
        """
        Inference-only mode: Load latest checkpoint and perform test evaluation.
        Skips all training.
        """
        print(f"\n{'='*80}")
        print("MINILM PHISHING DETECTION - INFERENCE MODE (CHECKPOINT RESUME)")
        print(f"{'='*80}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {Config.DEVICE}")
        print("="*80 + "\n")
        
        # Load test dataset
        _, _, test_dataset = self.load_datasets()
        test_loader = DataLoader(
            test_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=Config.NUM_WORKERS, 
            pin_memory=Config.PIN_MEMORY, 
            prefetch_factor=Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None
        )
        print(f"[OK] Test dataset loaded: {len(test_dataset):,} samples\n")
        
        # Find latest checkpoint
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
        
        # Load model state
        print("\nLoading model and checkpoint state...")
        model = self.create_model(vocab_size=len(self.tokenizer))
        criterion = FocalLoss().to(Config.DEVICE)
        
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=Config.DEVICE, weights_only=False)
            
            # Load model state dict
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.to(Config.DEVICE)
            print(f"[OK] Model state loaded")
            
            # Restore training history
            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']
                print(f"[OK] Training history restored ({len(self.training_history['train_losses'])} epochs)")
            
            checkpoint_epoch = checkpoint.get('epoch', 0)
            best_kpi_score = checkpoint.get('best_kpi_score', 0.0)
            
            print(f"[OK] Checkpoint epoch: {checkpoint_epoch}")
            print(f"[OK] Best KPI score at checkpoint: {best_kpi_score:.4f}\n")
            
        except Exception as e:
            print(f"[FAIL] Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Find best model epoch
        best_model_dirs = sorted(Config.SAVE_ROOT.glob("best_model_epoch_*"))
        if best_model_dirs:
            best_model_dir = best_model_dirs[-1]
            best_model_epoch = int(best_model_dir.name.split("_")[-1])
            print(f"[OK] Best model found at epoch: {best_model_epoch}")
        else:
            best_model_epoch = checkpoint_epoch
            print(f"[WARN] No best_model_epoch_* directory found, using checkpoint epoch: {best_model_epoch}")
        
        # Run evaluation
        print("="*60)
        print("TEST INFERENCE")
        print("="*60 + "\n")
        
        model.eval()
        kpi_compliance = self._evaluate_test_set(model, test_loader, criterion, best_model_epoch)
        
        print("\n" + "="*80)
        print("INFERENCE COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Checkpoint Used: {latest_checkpoint.name}")
        print(f"Checkpoint Epoch: {checkpoint_epoch}")
        print(f"KPI Compliance: {'[PASS] ACHIEVED' if kpi_compliance else '[WARN] PARTIAL'}")
        print(f"Results Directory: {Config.SAVE_ROOT / 'final_test_evaluation'}")
        print("="*80 + "\n")
        
        return kpi_compliance

    def _evaluate_test_set(self, model: nn.Module, test_loader: DataLoader, criterion: nn.Module, 
                            best_epoch: int, save_root: Path = None) -> bool:
        """Final evaluation on test set using production-ready merged model."""
        if save_root is None:
            save_root = Config.SAVE_ROOT
        test_inference_dir = save_root / "final_test_evaluation"
        test_inference_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("LOADING PRODUCTION MODEL FOR TEST INFERENCE")
        print(f"{'='*60}")
        
        # Load merged production model
        best_model_dir = save_root / f"best_model_epoch_{best_epoch:03d}"
        merged_model_path = best_model_dir / "model_merged_full.pt"
        
        if merged_model_path.exists():
            try:
                print(f"Loading merged model: {merged_model_path.name}")
                model = torch.load(merged_model_path, map_location=Config.DEVICE, weights_only=False)
                model.eval()
                print(f"[OK] Using production-ready merged model (LoRA weights integrated)")
            except Exception as e:
                print(f"[WARN] Failed to load merged model: {e}. Falling back.")
                model.eval()
        else:
            print(f"[WARN] Merged model not found. Using PyTorch model.")
            model.eval()
        
        print(f"Results will be saved to: {test_inference_dir.name}")
        print(f"{'='*60}\n")
        
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
        
        # Compute metrics
        test_loss = test_running_loss / len(test_loader.dataset)
        test_probs = np.array(all_probs)
        test_labels = np.array(all_labels)
        
        optimal_threshold = self.training_history['thresholds'][-1] if self.training_history['thresholds'] else 0.5
        test_preds = (test_probs >= optimal_threshold).astype(int)
        
        test_metrics = self.kpi_evaluator.evaluate_metrics(test_labels, test_preds, test_probs)
        test_metrics['test_loss'] = test_loss
        test_metrics['threshold_used'] = optimal_threshold
        test_metrics['model_used'] = 'model_merged_full.pt' if merged_model_path.exists() else 'model_full.pt'
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'url': all_urls,
            'true_label': test_labels,
            'predicted_label': test_preds,
            'prob_malicious': test_probs,
            'correct': test_labels == test_preds
        })
        predictions_df.to_csv(test_inference_dir / "test_predictions.csv", index=False)
        print(f"[OK] Predictions saved to: test_predictions.csv")
        
        # Save artifacts
        artifact_saver = ArtifactSaver(test_inference_dir)
        artifact_saver.save_test_metrics(test_metrics, optimal_threshold)
        artifact_saver.save_test_plots(test_labels, test_probs, optimal_threshold)
        
        # Print summary
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
        
        # Save results JSON
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
        
        print(f"\n[OK] Final results saved: final_results.json")
        
        return test_metrics['kpi_compliance']

    def onnx_inference(self, onnx_model_type: str = 'int8') -> bool:
        """
        ONNX Inference Mode: Load ONNX model and evaluate on test set.
        Inference runs entirely through ONNX Runtime.
        
        Args:
            onnx_model_type: 'int8' for quantized, 'fp32' for original, or custom path
        """
        print("\n" + "="*80)
        print("MINILM PHISHING DETECTION - ONNX INFERENCE MODE")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Runtime: ONNX Runtime (CPU-optimized)")
        print("="*80 + "\n")
        
        # Validate ONNX Runtime
        try:
            import onnxruntime as ort
            print(f"[OK] ONNX Runtime version: {ort.__version__}")
            available_providers = ort.get_available_providers()
            print(f"[OK] Available providers: {available_providers}")
        except ImportError:
            print("[FAIL] ERROR: onnxruntime not installed!")
            print("Install with: pip install onnxruntime")
            return False
        
        # Find ONNX model
        print(f"\n{'='*60}")
        print("ONNX MODEL SEARCH")
        print(f"{'='*60}")
        
        best_model_dirs = sorted(Config.SAVE_ROOT.glob("best_model_epoch_*"))
        
        if not best_model_dirs:
            print("[FAIL] ERROR: No best_model_epoch_* directory found!")
            print(f"Expected in: {Config.SAVE_ROOT}")
            print("Please run training first.")
            return False
        
        best_model_dir = best_model_dirs[-1]
        best_epoch = int(best_model_dir.name.split("_")[-1])
        print(f"[OK] Best model directory: {best_model_dir.name}")
        print(f"[OK] Best model epoch: {best_epoch}")
        
        onnx_quant_path = best_model_dir / "model_quant_8bit.onnx"
        onnx_fp32_path = best_model_dir / "model.onnx"
        
        # Determine model path
        if os.path.isfile(onnx_model_type):
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
            elif onnx_fp32_path.exists():
                onnx_model_path = onnx_fp32_path
                model_type = "FP32 (INT8 not available)"
                model_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
                print(f"[WARN] INT8 model not found, falling back to FP32: {onnx_fp32_path.name}")
                print(f"   Model size: {model_size:.2f} MB")
            else:
                print("[FAIL] ERROR: No ONNX model found!")
                return False
        
        print(f"{'='*60}\n")
        
        # Load datasets
        _, _, test_dataset = self.load_datasets()
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=False
        )
        print(f"[OK] Test dataset loaded: {len(test_dataset):,} samples\n")
        
        # Load threshold from deployment metadata
        optimal_threshold = None
        metadata_path = best_model_dir / "deployment_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                for key in ['threshold', 'optimal_threshold']:
                    if key in metadata and metadata[key] is not None:
                        optimal_threshold = float(metadata[key])
                        print(f"[OK] Loaded threshold from metadata: {optimal_threshold:.4f}")
                        break
            except Exception as e:
                print(f"[WARN] Could not load metadata: {e}")
                
        if optimal_threshold is None:
            optimal_threshold = 0.5
            print(f"[WARN] Using default threshold: {optimal_threshold:.4f}")
        
        # Create ONNX Session
        print(f"\n{'='*60}")
        print("ONNX SESSION INITIALIZATION")
        print(f"{'='*60}")
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = Config.NUM_WORKERS
        sess_options.inter_op_num_threads = 2
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        cuda_usable = False
        if 'CUDAExecutionProvider' in available_providers:
            try:
                if hasattr(ort, 'preload_dlls'):
                    ort.preload_dlls(cuda=True, cudnn=True)
                    print("[OK] Preloaded CUDA/cuDNN via ort.preload_dlls()")
                    cuda_usable = True
            except Exception as e:
                print(f"[WARN] Preloading failed: {e}")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda_usable else ['CPUExecutionProvider']
        ort.set_default_logger_severity(3)  # ERROR only
        
        try:
            session = ort.InferenceSession(str(onnx_model_path), sess_options=sess_options, providers=providers)
            ort.set_default_logger_severity(1)
            active_provider = session.get_providers()[0]
            execution_device = "CUDA (GPU)" if active_provider == 'CUDAExecutionProvider' else "CPU"
            print(f"[OK] ONNX session created successfully on {execution_device}")
            
            input_names = [inp.name for inp in session.get_inputs()]
            output_names = [out.name for out in session.get_outputs()]
        except Exception as e:
            ort.set_default_logger_severity(1)
            print(f"[FAIL] Failed to create ONNX session: {e}")
            return False
            
        print(f"{'='*60}\n")
        
        # Run ONNX inference
        print(f"{'='*60}")
        print("ONNX TEST INFERENCE")
        print(f"{'='*60}\n")
        
        all_probs, all_labels, all_urls = [], [], []
        inference_times = []
        
        total_start = time.perf_counter()
        
        for batch in tqdm(test_loader, desc="ONNX Inference"):
            input_ids = batch['input_ids'].numpy()
            attention_mask = batch['attention_mask'].numpy()
            labels = batch['labels'].numpy()
            
            ort_inputs = {
                input_names[0]: input_ids,
                input_names[1]: attention_mask
            }
            
            batch_start = time.perf_counter()
            ort_outputs = session.run(output_names, ort_inputs)
            batch_end = time.perf_counter()
            
            inference_times.append(batch_end - batch_start)
            
            logits = ort_outputs[0]
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            all_probs.extend(probs[:, 1])
            all_labels.extend(labels)
            all_urls.extend(batch['url'])
        
        total_end = time.perf_counter()
        total_inference_time = total_end - total_start
        
        # Compute metrics
        test_probs = np.array(all_probs)
        test_labels = np.array(all_labels)
        test_preds = (test_probs >= optimal_threshold).astype(int)
        
        test_metrics = self.kpi_evaluator.evaluate_metrics(test_labels, test_preds, test_probs)
        test_metrics['model_used'] = f'ONNX {model_type}'
        test_metrics['threshold_used'] = optimal_threshold
        
        total_samples = len(test_labels)
        avg_batch_time_ms = np.mean(inference_times) * 1000
        avg_sample_time_ms = (total_inference_time / total_samples) * 1000
        throughput = total_samples / total_inference_time
        
        # Save results
        onnx_results_dir = Config.SAVE_ROOT / "onnx_test_evaluation"
        onnx_results_dir.mkdir(parents=True, exist_ok=True)
        
        predictions_df = pd.DataFrame({
            'url': all_urls,
            'true_label': test_labels,
            'predicted_label': test_preds,
            'prob_malicious': test_probs,
            'correct': test_labels == test_preds
        })
        predictions_df.to_csv(onnx_results_dir / "onnx_test_predictions.csv", index=False)
        print(f"\n[OK] Predictions saved to: onnx_test_predictions.csv")
        
        # Print summary
        print(f"\n{'='*80}")
        print("ONNX INFERENCE RESULTS")
        print(f"{'='*80}")
        print(f"Model: {onnx_model_path.name} ({model_type})")
        print(f"Model Size: {model_size:.2f} MB {'[PASS]' if model_size <= Config.MAX_MODEL_SIZE_MB else '[FAIL]'}")
        print(f"Best Epoch: {best_epoch}")
        print(f"Threshold: {optimal_threshold:.4f}")
        print(f"\nClassification Metrics:")
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
        print(f"\nPerformance Benchmarks:")
        print(f"  Total inference time:    {total_inference_time:.2f}s")
        print(f"  Avg per-sample time:     {avg_sample_time_ms:.3f} ms")
        print(f"  Throughput:              {throughput:.0f} URLs/sec")
        print(f"  Execution device:        {execution_device}")
        
        full_compliance = test_metrics['kpi_compliance'] and (model_size <= Config.MAX_MODEL_SIZE_MB)
        print(f"\n  Overall: {'[PASS] ALL KPIs MET - PRODUCTION READY' if full_compliance else '[FAIL] KPIs NOT FULLY MET'}")
        print(f"{'='*80}")
        
        # Save ONNX final results
        onnx_final_results = {
            'test_metrics': test_metrics,
            'model_info': {
                'model_path': str(onnx_model_path),
                'model_type': model_type,
                'model_size_mb': model_size,
                'best_epoch': best_epoch,
                'execution_provider': active_provider,
            },
            'performance_benchmarks': {
                'total_inference_time_sec': total_inference_time,
                'total_samples': total_samples,
                'avg_sample_time_ms': avg_sample_time_ms,
                'throughput_urls_per_sec': throughput,
            },
            'kpi_compliance': full_compliance,
            'threshold': optimal_threshold,
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(Config.SAVE_ROOT / "onnx_inference_results.json", 'w') as f:
            json.dump(onnx_final_results, f, indent=2, default=str)
            
        return full_compliance


def main() -> bool:
    """Main CLI entry point for inference."""
    parser = argparse.ArgumentParser(
        description="MiniLM Phishing URL Detection - Modular Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='4_config.yaml',
        help="Path to centralized configuration file (default: 4_config.yaml)"
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['inference', 'onnx-inference'],
        default='inference',
        help="""
        Inference mode:
        - 'inference': Load latest PyTorch checkpoint / merged model
        - 'onnx-inference': Load ONNX model and run CPU-optimized evaluation
        """
    )
    
    parser.add_argument(
        '--onnx-model',
        type=str,
        default='int8',
        help="""
        ONNX model variant for onnx-inference mode:
        - 'int8': INT8 quantized model (model_quant_8bit.onnx)
        - 'fp32': FP32 original model (model.onnx)
        - '/path/to/model.onnx': Custom ONNX model path
        (default: int8)
        """
    )
    
    args = parser.parse_args()
    
    # Load configuration
    Config.load_from_yaml(args.config)
    
    try:
        if args.mode == 'inference':
            inference = PhishingDetectionInference()
            success = inference.inference_from_checkpoint()
        elif args.mode == 'onnx-inference':
            inference = PhishingDetectionInference()
            success = inference.onnx_inference(onnx_model_type=args.onnx_model)
        else:
            success = False
            
        return success
        
    except KeyboardInterrupt:
        print("\n\n[WARN] Inference interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n[FAIL] ERROR during inference: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
