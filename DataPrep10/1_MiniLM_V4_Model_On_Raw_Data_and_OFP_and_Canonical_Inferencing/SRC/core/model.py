from typing import Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from core.config import Config


class MiniLMURLClassifier(nn.Module):
    """MiniLM-L12-H384 classifier optimized for URL phishing detection."""
    
    def __init__(self, vocab_size: Optional[int] = None):
        super().__init__()
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.encoder = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)
        if vocab_size is not None and vocab_size != self.config.vocab_size:
            print(f"[MODEL] Resizing base model token embeddings from {self.config.vocab_size} to {vocab_size}")
            self.encoder.resize_token_embeddings(vocab_size)
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
        
        # Stability check (tracer-friendly unconditional clamping)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return logits


def apply_structured_pruning(model: nn.Module, amount: float = Config.PRUNING_RATIO) -> None:
    """Pruning disabled/enabled based on configuration."""
    if amount <= 0.0:
        print("Pruning disabled (MiniLM already compact)")
        return
    
    # Optional structured pruning logic can be implemented here if amount > 0.0
    import torch.nn.utils.prune as prune
    print(f"Applying structured pruning with ratio: {amount}")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')


def save_model_summary(model: nn.Module, input_size: Tuple[int, int], save_path: str = "model_summery.txt") -> None:
    """
    Save comprehensive model summary to a text file.
    
    This simplified version doesn't use torchinfo, making it more reliable
    for models with multiple inputs like MiniLM.
    """
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
        print(f"[ERROR] Failed to save model summary: {e}")
        import traceback
        traceback.print_exc()
