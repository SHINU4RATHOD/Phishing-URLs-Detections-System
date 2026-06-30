import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Ensure sibling packages under SRC can be imported regardless of run directory
sys.path.append(str(Path(__file__).resolve().parent))

from core.config import Config
from core.dataset import URLDataset, create_weighted_sampler
from core.loss import FocalLoss
from core.model import MiniLMURLClassifier, apply_structured_pruning
from core.evaluator import EnhancedKPIEvaluator
from core.utils import CheckpointManager, ArtifactSaver, ModelExporter
from core.mlflow_logger import MLflowManager


class PhishingDetectionTrainer:
    """Main training orchestrator for URL Phishing Detection."""
    
    def __init__(self):
        Config.setup_reproducibility()
        Config.setup_paths()
        
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.checkpoint_manager = CheckpointManager(Config.CHECKPOINT_DIR)
        self.kpi_evaluator = EnhancedKPIEvaluator()
        self.mlflow_manager = MLflowManager()
        self.mlflow_run_id = None
        self.test_metrics_to_log = None
        
        self.training_history = {
            'train_losses': [], 'val_losses': [],
            'train_accs': [], 'val_accs': [],
            'kpi_scores': [], 'thresholds': []
        }
        self.device_type = 'cuda' if 'cuda' in str(Config.DEVICE) else 'cpu'
    
    def load_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load datasets as DataFrames to allow dynamic Stratified K-Fold splitting."""
        print("\n" + "="*60)
        print("LOADING DATASETS")
        print("="*60)
        
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
        
        print(f"\nDataset Sizes:")
        print(f"  Train:      {len(train_df):,}")
        print(f"  Validation: {len(val_df):,}")
        print(f"  Test:       {len(test_df):,}")
        print(f"  Total:      {len(train_df) + len(val_df) + len(test_df):,}")
        
        return train_df, val_df, test_df

    def train_custom_tokenizer(self, train_df: pd.DataFrame) -> None:
        """Train and save a custom Byte-Level BPE tokenizer on training URLs."""
        print("\n" + "="*60)
        print("TRAINING CUSTOM BYTE-LEVEL BPE TOKENIZER")
        print("="*60)
        
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.processors import TemplateProcessing
        from transformers import PreTrainedTokenizerFast
        
        # Train BPE
        raw_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        raw_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        
        trainer = BpeTrainer(
            vocab_size=Config.CUSTOM_TOKENIZER_VOCAB_SIZE,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        )
        
        urls = train_df['input'].astype(str).tolist()
        def batch_iterator():
            for i in range(0, len(urls), 1000):
                yield urls[i:i+1000]
                
        raw_tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
        
        # Setup template processing to handle special tokens
        raw_tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", raw_tokenizer.token_to_id("[CLS]")),
                ("[SEP]", raw_tokenizer.token_to_id("[SEP]")),
            ],
        )
        
        # Save tokenizer
        save_path = Path(Config.CUSTOM_TOKENIZER_PATH)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        raw_tokenizer.save(str(save_path))
        print(f"[OK] Custom tokenizer trained and saved to: {save_path}")
        
        # Load as PreTrainedTokenizerFast
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(save_path),
            unk_token="[UNK]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            mask_token="[MASK]"
        )
        print(f"[OK] Loaded custom tokenizer from file, vocab size: {len(self.tokenizer)}")

    def train_custom_unigram_tokenizer(self, train_df: pd.DataFrame) -> None:
        """Train and save a custom SentencePiece Unigram tokenizer on training URLs."""
        print("\n" + "="*60)
        print("TRAINING CUSTOM SENTENCEPIECE UNIGRAM TOKENIZER")
        print("="*60)
        
        from tokenizers import Tokenizer
        from tokenizers.models import Unigram
        from tokenizers.trainers import UnigramTrainer
        from tokenizers.pre_tokenizers import Metaspace
        from tokenizers.processors import TemplateProcessing
        from transformers import PreTrainedTokenizerFast
        
        raw_tokenizer = Tokenizer(Unigram())
        raw_tokenizer.pre_tokenizer = Metaspace()
        
        trainer = UnigramTrainer(
            vocab_size=Config.CUSTOM_TOKENIZER_VOCAB_SIZE,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
            unk_token="[UNK]"
        )
        
        urls = train_df['input'].astype(str).tolist()
        def batch_iterator():
            for i in range(0, len(urls), 1000):
                yield urls[i:i+1000]
                
        raw_tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
        
        # Setup template processing to handle special tokens
        raw_tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", raw_tokenizer.token_to_id("[CLS]")),
                ("[SEP]", raw_tokenizer.token_to_id("[SEP]")),
            ],
        )
        
        # Save tokenizer
        save_path = Path(Config.CUSTOM_TOKENIZER_PATH)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        raw_tokenizer.save(str(save_path))
        print(f"[OK] Custom Unigram tokenizer trained and saved to: {save_path}")
        
        # Load as PreTrainedTokenizerFast
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(save_path),
            unk_token="[UNK]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            mask_token="[MASK]"
        )
        print(f"[OK] Loaded custom tokenizer from file, vocab size: {len(self.tokenizer)}")
    
    def create_model(self, vocab_size: Optional[int] = None) -> nn.Module:
        """Build MiniLM-L12-H384 model with LoRA."""
        print("\n" + "="*60)
        print("BUILDING MODEL")
        print("="*60)
        
        base_model = MiniLMURLClassifier(vocab_size=vocab_size)
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
        print(f"Target: <{Config.MAX_MODEL_SIZE_MB}MB model with {Config.TARGET_ACCURACY:.0%} accuracy")
        print("="*80)
        
        train_df, val_df, test_df = self.load_dataframes()
        
        # Train custom tokenizer if enabled
        if Config.USE_CUSTOM_TOKENIZER_BYTE_LEVEL_BPE:
            self.train_custom_tokenizer(train_df)
        elif Config.USE_CUSTOM_TOKENIZER_SENTENCEPIECE_UNIGRAM_BPE:
            self.train_custom_unigram_tokenizer(train_df)
            
        test_dataset = URLDataset(test_df, self.tokenizer)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=Config.PIN_MEMORY, 
            prefetch_factor=2 if 2 > 0 else None
        )
        
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
            
            model = self.create_model(vocab_size=len(self.tokenizer))
            
            if Config.USE_WEIGHTED_SAMPLING:
                sampler = create_weighted_sampler(train_dataset.labels)
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=Config.BATCH_SIZE, 
                    sampler=sampler, 
                    num_workers=Config.NUM_WORKERS, 
                    pin_memory=Config.PIN_MEMORY, 
                    prefetch_factor=Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None
                )
                print(f"[OK] Training with weighted sampling (balanced batches)")
            else:
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=Config.BATCH_SIZE, 
                    shuffle=True, 
                    num_workers=Config.NUM_WORKERS, 
                    pin_memory=Config.PIN_MEMORY, 
                    prefetch_factor=Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None
                )
                print(f"[OK] Training with standard random shuffling")
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=Config.BATCH_SIZE, 
                shuffle=False, 
                num_workers=2, 
                pin_memory=Config.PIN_MEMORY, 
                prefetch_factor=2 if 2 > 0 else None
            )
            
            optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY, eps=1e-8)
            total_steps = len(train_loader) * Config.NUM_EPOCHS
            warmup_steps = int(Config.LR_WARMUP_RATIO * total_steps)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=warmup_steps, 
                num_training_steps=total_steps, 
                num_cycles=0.5, 
                last_epoch=-1
            )
            criterion = FocalLoss().to(Config.DEVICE)
            scaler = GradScaler(self.device_type, enabled=Config.USE_AMP)
            
            start_epoch = 1
            best_kpi_score = 0.0
            best_model_epoch = 0
            patience_counter = 0
            optimal_threshold = 0.5
            
            latest_checkpoint = self.checkpoint_manager.find_latest_checkpoint()
            
            if latest_checkpoint:
                if Config.USE_STRATIFIED_KFOLD:
                    print(f"\n[RESUME] CHECKPOINT FOUND for Fold {fold_idx+1}: {latest_checkpoint.name}")
                else:
                    print(f"\n[RESUME] CHECKPOINT FOUND: {latest_checkpoint.name}")
                start_epoch, last_metrics, best_kpi_score, loaded_history, loaded_mlflow_run_id = \
                    self.checkpoint_manager.load_checkpoint(
                        latest_checkpoint, model, optimizer, scheduler, scaler
                    )
                if loaded_history:
                    self.training_history = loaded_history
                    if self.training_history.get('thresholds'):
                        optimal_threshold = self.training_history['thresholds'][-1]
                if loaded_mlflow_run_id:
                    self.mlflow_run_id = loaded_mlflow_run_id
                
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
            
            # Initialize MLOps MLflow Tracking
            self.mlflow_run_id = self.mlflow_manager.start_run(run_id=self.mlflow_run_id)

            for epoch in range(start_epoch, Config.NUM_EPOCHS + 1):
                print(f"\n{'='*60}")
                if Config.USE_STRATIFIED_KFOLD:
                    print(f"FOLD {fold_idx+1} - EPOCH {epoch}/{Config.NUM_EPOCHS}")
                else:
                    print(f"EPOCH {epoch}/{Config.NUM_EPOCHS}")
                print(f"{'='*60}")
                
                train_loss, train_acc = self._train_epoch(
                    model, train_loader, optimizer, scheduler, criterion, scaler, epoch
                )
                val_loss, val_probs, val_labels = self._validate_epoch(model, val_loader, criterion)
                
                optimal_threshold, _threshold_info = self.kpi_evaluator.find_optimal_threshold_strict(val_labels, val_probs)
                val_preds = (val_probs >= optimal_threshold).astype(int)
                val_metrics = self.kpi_evaluator.evaluate_metrics(val_labels, val_preds, val_probs)
                
                self._update_history(train_loss, train_acc, val_loss, val_metrics, optimal_threshold)
                self._print_epoch_summary(epoch, train_loss, train_acc, val_loss, val_metrics, optimal_threshold)
                
                # Log epoch metrics to MLflow cleanly
                self.mlflow_manager.log_epoch_metrics(
                    epoch, train_loss, train_acc, val_loss, val_metrics, optimal_threshold
                )

                if val_metrics['kpi_score'] > best_kpi_score:
                    best_kpi_score = val_metrics['kpi_score']
                    best_model_epoch = epoch
                    patience_counter = 0
                    self._save_best_model(model, epoch, val_metrics, optimal_threshold, fold_save_root)
                    if Config.USE_STRATIFIED_KFOLD:
                        print(f"[BEST] New best model for Fold {fold_idx+1}! KPI Score improved to {best_kpi_score:.4f}")
                    else:
                        print(f"[BEST] New best model! KPI Score improved to {best_kpi_score:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= Config.PATIENCE:
                        print(f"\nEarly stopping triggered at epoch {epoch}")
                        break
                
                # Save checkpoint with mlflow_run_id for metrics resumption
                self.checkpoint_manager.save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, val_metrics, 
                    optimal_threshold, best_kpi_score, self.training_history,
                    mlflow_run_id=self.mlflow_run_id
                )
                self.checkpoint_manager.cleanup_old_checkpoints(keep_last_n=3)
            
            if best_kpi_score > overall_best_kpi:
                overall_best_kpi = best_kpi_score
                overall_best_fold = fold_idx + 1
                
            print("\n" + "="*80)
            if Config.USE_STRATIFIED_KFOLD:
                print(f"FINAL TEST EVALUATION FOR FOLD {fold_idx+1}")
            else:
                print("FINAL TEST EVALUATION")
            print("="*80)
            fold_kpi_compliance = self._evaluate_test_set(model, test_loader, criterion, best_model_epoch, fold_save_root)
            if fold_kpi_compliance and overall_best_fold == fold_idx + 1:
                overall_kpi_compliance = True
                
            # Log final test evaluation metrics to MLflow
            if self.test_metrics_to_log:
                self.mlflow_manager.log_test_evaluation(
                    self.test_metrics_to_log, best_model_epoch, optimal_threshold
                )
                
            # Log best model checkpoints and artifacts
            best_epoch_dir = fold_save_root / f"best_model_epoch_{best_model_epoch:03d}"
            if best_epoch_dir.exists():
                self.mlflow_manager.log_artifacts(best_epoch_dir)
                
            # Log test evaluation prediction sheets and curves
            test_eval_dir = fold_save_root / "final_test_evaluation"
            if test_eval_dir.exists():
                self.mlflow_manager.log_artifacts(test_eval_dir, artifact_path="model_output_artifacts/final_test_evaluation")
                
            # Automatically register finalized production model version
            if fold_kpi_compliance and Config.MLFLOW_REGISTER_MODEL and self.mlflow_run_id:
                model_uri = f"runs:/{self.mlflow_run_id}/model_output_artifacts/model_merged_full.pt"
                self.mlflow_manager.register_model_version(
                    Config.MLFLOW_MODEL_NAME, model_uri,
                    description=f"Merged Production-grade MiniLM URL Classifier (Accuracy: {self.test_metrics_to_log.get('accuracy', 0.0):.4f})"
                )
                
            # Terminate active MLflow run session cleanly
            self.mlflow_manager.end_run()
                
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if Config.USE_STRATIFIED_KFOLD:
            print(f"Overall Best Model: Fold {overall_best_fold} (KPI Score: {overall_best_kpi:.4f})")
        else:
            print(f"Overall Best Model (KPI Score: {overall_best_kpi:.4f})")
        print(f"KPI Compliance: {'[PASS] ACHIEVED' if overall_kpi_compliance else '[WARN] PARTIAL'}")
        print("="*80 + "\n")
        
        return overall_kpi_compliance
        
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, 
                     scheduler: Any, criterion: nn.Module, scaler: GradScaler, epoch: int) -> Tuple[float, float]:
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
            with autocast(self.device_type, enabled=Config.USE_AMP):
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
            if Config.CLASSIFICATION_LAYER_TYPE == "sigmoid":
                preds = (torch.sigmoid(logits).squeeze(-1) >= 0.5).cpu().numpy().astype(int)
            else:
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
    
    def _validate_epoch(self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, np.ndarray, np.ndarray]:
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
                if Config.CLASSIFICATION_LAYER_TYPE == "sigmoid":
                    probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                    all_probs.extend(probs)
                else:
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    all_probs.extend(probs[:, 1])
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader.dataset) if len(all_probs) > 0 else float('inf')
        return epoch_loss, np.array(all_probs), np.array(all_labels)
    
    def _update_history(self, train_loss: float, train_acc: float, val_loss: float, val_metrics: Dict, threshold: float) -> None:
        """Update training history."""
        self.training_history['train_losses'].append(train_loss)
        self.training_history['val_losses'].append(val_loss)
        self.training_history['train_accs'].append(train_acc)
        self.training_history['val_accs'].append(val_metrics['accuracy'])
        self.training_history['kpi_scores'].append(val_metrics['kpi_score'])
        self.training_history['thresholds'].append(threshold)
    
    def _print_epoch_summary(self, epoch: int, train_loss: float, train_acc: float, val_loss: float, val_metrics: Dict, threshold: float) -> None:
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
    
    def _save_best_model(self, model: nn.Module, epoch: int, metrics: Dict, threshold: float, save_root: Path = None) -> None:
        """Save best model with all exports and artifacts."""
        if save_root is None:
            save_root = Config.SAVE_ROOT
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
            print(f"[WARN] Merged model not found. Using training model.")
            model.eval()
        
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
                if Config.CLASSIFICATION_LAYER_TYPE == "sigmoid":
                    probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                    all_probs.extend(probs)
                else:
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
        
        # Save metrics state locally for trainer logger mapping
        self.test_metrics_to_log = test_metrics
        
        return test_metrics['kpi_compliance']


def main() -> bool:
    """Main CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description="MiniLM Phishing URL Detection - Modular Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='4_config.yaml',
        help="Path to centralized configuration file (default: 4_config.yaml)"
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help="Override number of training epochs in configuration file"
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help="Override training batch size in configuration file"
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help="Override learning rate in configuration file"
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        default=False,
        help="Enable interactive prompt warnings"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    Config.load_from_yaml(args.config)
    
    # Apply CLI overrides
    if args.epochs is not None:
        Config.NUM_EPOCHS = args.epochs
        print(f"[CLI] Overriding NUM_EPOCHS to {args.epochs}")
    
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
        print(f"[CLI] Overriding BATCH_SIZE to {args.batch_size}")
    
    if args.lr is not None:
        Config.LR = args.lr
        print(f"[CLI] Overriding LR to {args.lr}")
        
    try:
        print("\n" + "="*80)
        print(" " * 20 + "MiniLM PHISHING URL DETECTION")
        print(" " * 15 + "Modular Production-Grade Training Pipeline")
        print("="*80)
        print(f"Mode:                  TRAINING")
        print(f"Config File:           {args.config}")
        print(f"Target Model Size:     <{Config.MAX_MODEL_SIZE_MB}MB with {Config.TARGET_ACCURACY:.0%} accuracy")
        print(f"Architecture:          MiniLM v3 Base + LoRA + Focal Loss")
        print(f"Device:                {Config.DEVICE}")
        print(f"Interactive Mode:      {'Enabled' if args.interactive else 'Disabled'}")
        print("="*80 + "\n")
        
        trainer = PhishingDetectionTrainer()
        success = trainer.train()
        return success
    
    except KeyboardInterrupt:
        print("\n\n[WARN] Training interrupted by user")
        return False
    
    except Exception as e:
        print(f"\n[FAIL] ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
