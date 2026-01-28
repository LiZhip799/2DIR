import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedL1Loss(nn.Module):
    """L1 Loss ignoring padded regions in distance maps [cite: 61, 64]"""
    def forward(self, inputs, targets):
        mask = (targets != 0)
        loss = torch.abs(inputs - targets)
        return loss[mask].mean() if loss[mask].numel() > 0 else torch.tensor(0.0, device=inputs.device)

def get_prop_loss(pred_ss, target_ss, pred_props, target_props, prop_tasks):
    """Multitask loss for properties"""
    loss_ss = F.mse_loss(pred_ss, target_ss)
    loss_prop = sum(F.mse_loss(pred_props[t], target_props[t]) for t in prop_tasks)
    return loss_ss + loss_prop