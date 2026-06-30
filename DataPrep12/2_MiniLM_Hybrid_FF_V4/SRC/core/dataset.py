from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from core.config import Config


class HybridURLDataset(Dataset):
    """
    PyTorch Dataset for hybrid GLU Fusion model.
    
    Reads hybrid CSV containing:
      - input:              URL text → tokenize with HuggingFace
      - label:              binary 0/1 threat labels
      - h_* numeric:        Continuous features → Z-score normalized
      - h_* binary:         Binary flags → passed through
      - hF_* flags:         Category flags → passed through
      - h_primary_category: Leaky categories → dropped
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        tokenizer,
        feature_cols: List[str],
        norm_stats: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.tokenizer = tokenizer
        self.urls = df['input'].astype(str).tolist()
        self.labels = df['label'].astype(int).tolist()
        
        # Extract metadata for Samsung Decision Engine Layer
        self.metadata = {
            'h_severity_score': df['severity_score'].fillna(0).astype(float).tolist() if 'severity_score' in df.columns else (df['h_severity_score'].fillna(0).astype(float).tolist() if 'h_severity_score' in df.columns else [0.0]*len(self.urls)),
            'h_flags_count': df['flags_count'].fillna(0).astype(int).tolist() if 'flags_count' in df.columns else (df['h_flags_count'].fillna(0).astype(int).tolist() if 'h_flags_count' in df.columns else [0]*len(self.urls)),
            'h_primary_category': df['primary_category'].fillna('UNKNOWN').astype(str).tolist() if 'primary_category' in df.columns else (df['h_primary_category'].fillna('UNKNOWN').astype(str).tolist() if 'h_primary_category' in df.columns else ['UNKNOWN']*len(self.urls))
        }
        
        # Extract heuristic features as numpy array
        self.feature_cols = feature_cols
        
        # Handle cases where feature_cols is empty (prevent Z-score normalization crash)
        if feature_cols:
            features_df = df[feature_cols].fillna(0).astype(np.float32)
            
            # Apply Z-score normalization to NUMERIC columns only (not binary/flags)
            self.norm_stats = norm_stats
            if norm_stats is not None:
                for col in Config.NUMERIC_FEATURE_COLS:
                    if col in features_df.columns:
                        mean, std = norm_stats[col]
                        features_df[col] = (features_df[col] - mean) / (std + 1e-8)
            
            self.features = features_df.values  # (N, feature_dim)
        else:
            self.features = np.zeros((len(self.urls), Config.HEURISTIC_DIM), dtype=np.float32)
        
        print(f"Dataset: {len(self.urls):,} samples | {len(feature_cols)} heuristic features detected")
        label_dist = pd.Series(self.labels).value_counts().to_dict()
        print(f"Label distribution: {label_dist}")
    
    def __len__(self) -> int:
        return len(self.urls)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        url = self.urls[idx]
        label = self.labels[idx]
        heuristic = self.features[idx]
        
        # Pads/truncates features array to align with Config.HEURISTIC_DIM
        if len(heuristic) < Config.HEURISTIC_DIM:
            padded = np.zeros(Config.HEURISTIC_DIM, dtype=np.float32)
            padded[:len(heuristic)] = heuristic
            heuristic = padded
        elif len(heuristic) > Config.HEURISTIC_DIM:
            heuristic = heuristic[:Config.HEURISTIC_DIM]
            
        encoding = self.tokenizer(
            url, add_special_tokens=True, max_length=Config.MAX_LEN,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'heuristic_features': torch.tensor(heuristic, dtype=torch.float32),
            'labels': torch.tensor(label, dtype=torch.long),
            'url': url,
            'h_severity_score': float(self.metadata['h_severity_score'][idx]),
            'h_flags_count': int(self.metadata['h_flags_count'][idx]),
            'h_primary_category': str(self.metadata['h_primary_category'][idx])
        }
    
    @staticmethod
    def compute_normalization_stats(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """Compute mean/std for numeric columns from training data only."""
        stats = {}
        for col in Config.NUMERIC_FEATURE_COLS:
            if col in df.columns:
                vals = df[col].fillna(0).astype(np.float32)
                std_val = float(vals.std())
                stats[col] = (float(vals.mean()), std_val if std_val > 0 else 1.0)
        print(f"[OK] Normalization Z-score statistics calculated for {len(stats)} features.")
        return stats
    
    @staticmethod
    def detect_feature_columns(df: pd.DataFrame) -> List[str]:
        """Auto-detect heuristic feature columns from hybrid CSV."""
        feature_cols = []
        # Numeric h_ features
        for col in Config.NUMERIC_FEATURE_COLS:
            if col in df.columns:
                feature_cols.append(col)
        # Binary h_ features
        for col in Config.BINARY_FEATURE_COLS:
            if col in df.columns:
                feature_cols.append(col)
        # hF_* flags (auto-detected)
        hf_cols = sorted([c for c in df.columns if c.startswith('hF_')])
        feature_cols.extend(hf_cols)
        
        # Drop leaky categorical categories
        feature_cols = [c for c in feature_cols if c not in Config.DROP_COLS]
        
        print(f"[OK] Detected {len(feature_cols)} heuristic columns:")
        print(f"  Numeric: {len([c for c in feature_cols if c in Config.NUMERIC_FEATURE_COLS])}")
        print(f"  Binary:  {len([c for c in feature_cols if c in Config.BINARY_FEATURE_COLS])}")
        print(f"  Flags:   {len(hf_cols)}")
        return feature_cols


def create_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """Create WeightedRandomSampler to balance train batches."""
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    class_weights = {
        class_id: total_samples / count 
        for class_id, count in class_counts.items()
    }
    
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    print(f"\n{'='*60}")
    print("WEIGHTED BALANCED SAMPLER CREATED")
    print(f"{'='*60}")
    for class_id, count in sorted(class_counts.items()):
        percentage = (count / total_samples) * 100
        weight = class_weights[class_id]
        label_name = "Benign" if class_id == 0 else "Phishing"
        print(f"  {label_name:10} ({class_id}): {count:,} samples ({percentage:.2f}%) - Weight: {weight:.4f}")
    print(f"{'='*60}\n")
    
    return sampler
