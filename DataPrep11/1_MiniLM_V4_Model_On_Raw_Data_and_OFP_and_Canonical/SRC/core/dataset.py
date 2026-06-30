from collections import Counter
from typing import Dict, List
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from core.config import Config


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
        
        encoding = self.tokenizer(
            url, 
            add_special_tokens=True, 
            max_length=Config.MAX_LEN, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'url': url
        }


def create_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler based on inverse class frequencies to handle imbalanced data."""
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
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
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
