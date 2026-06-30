import torch
import torch.nn as nn
from core.config import Config


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
        if Config.CLASSIFICATION_LAYER_TYPE == "sigmoid":
            ce_loss = nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(-1), 
                targets.float(), 
                reduction='none'
            )
        else:
            ce_loss = nn.functional.cross_entropy(
                logits, 
                targets, 
                reduction='none', 
                label_smoothing=self.label_smoothing
            )
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
            if Config.CLASSIFICATION_LAYER_TYPE == "sigmoid":
                return nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), targets.float())
            else:
                return nn.functional.cross_entropy(logits, targets)
        
        return loss
