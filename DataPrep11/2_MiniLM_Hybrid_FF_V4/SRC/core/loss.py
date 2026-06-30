import torch
import torch.nn as nn
from typing import Optional, List

from core.config import Config


class FocalLoss(nn.Module):
    """
    Focal Loss with numerical stability.
    
     mines hard examples dynamically and balances minority threat classes.
    """
    
    def __init__(self, gamma: Optional[float] = None, alpha: Optional[list] = None, label_smoothing: Optional[float] = None):
        super().__init__()
        self.gamma = gamma if gamma is not None else Config.FOCAL_GAMMA
        self.label_smoothing = label_smoothing if label_smoothing is not None else Config.LABEL_SMOOTHING
        
        alpha_val = alpha if alpha is not None else Config.FOCAL_ALPHA
        if alpha_val:
            self.register_buffer('alpha_tensor', torch.tensor(alpha_val, dtype=torch.float))
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
        
        if self.alpha_tensor is not None:
            alpha_t = self.alpha_tensor.to(targets.device)[targets]
            focal_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss
        else:
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        loss = focal_loss.mean()
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("[WARN] NaN/Inf detected in loss, using CE fallback")
            if Config.CLASSIFICATION_LAYER_TYPE == "sigmoid":
                return nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), targets.float())
            else:
                return nn.functional.cross_entropy(logits, targets)
        
        return loss
