from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoConfig

from core.config import Config


class HeuristicMLP(nn.Module):
    """
    MLP tower for heuristic features → compact embedding.
    
    Architecture:  90 → 256 → LayerNorm → GELU → Dropout → 192 → LayerNorm → GELU
    """
    
    def __init__(
        self,
        input_dim: int = Config.HEURISTIC_DIM,
        hidden_dim: int = Config.HEURISTIC_MLP_HIDDEN,
        output_dim: int = Config.HEURISTIC_MLP_OUTPUT,
        dropout: float = Config.DROPOUT,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        self._init_weights()
    
    def _init_weights(self) -> None:
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GLUGate(nn.Module):
    """
    Gated Linear Unit for fusing text (384-dim) + heuristic (192-dim) embeddings.
    
    Mechanism:
        concat = [text_emb ; feat_emb]          # (batch, 384 + 192 = 576)
        gate   = σ(W_gate · concat + b_gate)     # (batch, hidden_dim = 384)
        value  = tanh(W_val · concat + b_val)    # (batch, hidden_dim = 384)
        output = gate ⊙ value                     # (batch, hidden_dim = 384)
    """
    
    def __init__(
        self,
        text_dim: int = Config.TEXT_EMBED_DIM,
        feat_dim: int = Config.HEURISTIC_MLP_OUTPUT,
        hidden_dim: int = Config.GLU_HIDDEN,
    ):
        super().__init__()
        concat_dim = text_dim + feat_dim  # 384 + 192 = 576
        self.gate_proj = nn.Linear(concat_dim, hidden_dim)
        self.value_proj = nn.Linear(concat_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self._init_weights()
    
    def _init_weights(self) -> None:
        for proj in [self.gate_proj, self.value_proj]:
            nn.init.xavier_normal_(proj.weight, gain=0.02)
            nn.init.zeros_(proj.bias)
    
    def forward(self, text_emb: torch.Tensor, feat_emb: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([text_emb, feat_emb], dim=-1)  # (batch, 576)
        gate = torch.sigmoid(self.gate_proj(concat))        # (batch, 384)
        value = torch.tanh(self.value_proj(concat))          # (batch, 384)
        fused = gate * value                                  # (batch, 384)
        return self.layer_norm(fused)


class HybridGLUClassifier(nn.Module):
    """
    Dual-tower GLU Fusion classifier for phishing URL detection.
    
    Tower 1 (Text):       input → MiniLM-L12 + LoRA → CLS → 384-dim
    Tower 2 (Heuristic):  90 features → MLP → 192-dim
    Fusion:               [384; 192] → GLU Gate → 384-dim
    Head:                 384 → 192 → 64 → 2 (binary classification logits)
    """
    
    def __init__(self, vocab_size: Optional[int] = None):
        super().__init__()
        
        # --- Tower 1: MiniLM Text Encoder ---
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.encoder = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)
        if vocab_size is not None and vocab_size != self.config.vocab_size:
            print(f"[MODEL] Resizing base model token embeddings from {self.config.vocab_size} to {vocab_size}")
            self.encoder.resize_token_embeddings(vocab_size)
        self.hidden_size = self.config.hidden_size  # 384
        
        # --- Tower 2: Heuristic MLP ---
        self.heuristic_mlp = HeuristicMLP()
        
        # --- GLU Fusion Gate ---
        self.glu_gate = GLUGate()
        
        # --- Classification Head (post-fusion) ---
        layers = []
        in_dim = Config.GLU_HIDDEN  # 384 (GLU output)
        for out_dim in Config.CLASSIFIER_DIMS:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(Config.DROPOUT),
            ])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, Config.NUM_CLASSES))
        self.classifier = nn.Sequential(*layers)
        self._init_classifier_weights()
    
    def _init_classifier_weights(self) -> None:
        """Xavier initialization for classifier layers."""
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        heuristic_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # --- Tower 1: Text encoding ---
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = outputs.last_hidden_state[:, 0]  # CLS token → (batch, 384)
        
        if heuristic_features is not None:
            # Pads/truncates features dynamically to align with Config.HEURISTIC_DIM
            if heuristic_features.size(1) < Config.HEURISTIC_DIM:
                padded = torch.zeros(heuristic_features.size(0), Config.HEURISTIC_DIM, device=heuristic_features.device, dtype=heuristic_features.dtype)
                padded[:, :heuristic_features.size(1)] = heuristic_features
                heuristic_features = padded
            elif heuristic_features.size(1) > Config.HEURISTIC_DIM:
                heuristic_features = heuristic_features[:, :Config.HEURISTIC_DIM]
                
            # --- Tower 2: Heuristic MLP ---
            feat_emb = self.heuristic_mlp(heuristic_features)  # (batch, 192)
            
            # --- GLU Fusion ---
            fused = self.glu_gate(text_emb, feat_emb)  # (batch, 384)
        else:
            # Fallback: text-only (pad with zeros for missing heuristic features)
            dummy_feat = torch.zeros(
                text_emb.size(0), Config.HEURISTIC_MLP_OUTPUT, 
                device=text_emb.device, dtype=text_emb.dtype
            )
            fused = self.glu_gate(text_emb, dummy_feat)
        
        logits = self.classifier(fused)  # (batch, 2)
        
        # Stability check (crucial for FP16 and AMP scaling)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return logits


class ModelCalibrator(nn.Module):
    """
    Learns a single scalar temperature T that scales logits (logits / T) 
    to produce mathematically calibrated probabilities (minimizing NLL).
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def calibrate(self, valid_logits: torch.Tensor, valid_labels: torch.Tensor):
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_fn():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(valid_logits), valid_labels)
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        return self.temperature.item()


def apply_structured_pruning(model: nn.Module, amount: float = Config.PRUNING_RATIO) -> None:
    """Structured pruning helper (pruning is currently disabled for MiniLM base)."""
    if amount <= 0.0:
        print("[OK] Pruning disabled (MiniLM already compact)")
        return


def save_model_summary(model: nn.Module, input_size: Tuple[int, int], save_path: str = "model_summery.txt") -> None:
    """Save comprehensive, dynamic model structure summary to a text file."""
    try:
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("MODEL SUMMARY: MiniLM Hybrid GLU URL Classifier")
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
        
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(summary_str)
        
        print(f"[OK] Model summary saved to {save_path}")
        print(f"  Total params: {total_params:,} | Trainable: {trainable_params:,} | Size: {size_mb:.2f} MB")
        
    except Exception as e:
        print(f"[WARN] Failed to save model summary: {e}")
