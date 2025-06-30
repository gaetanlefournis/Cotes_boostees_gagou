import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tools_ai_model import func_profit_loss


class ProfitAwareLoss(nn.Module):
    """Profit-aware loss function combining cross-entropy and profit loss, adding a penalty for small amount of predictions."""
    def __init__(
        self,
        coeff_ce_loss : float = 0.5,
        coeff_profit_loss : float = 0.5,
        coeff_small_preds_loss : float = 0.5,
        weights : torch.Tensor = None
    ):
        super().__init__()
        self.coeff_ce_loss = coeff_ce_loss
        self.coeff_profit_loss = coeff_profit_loss
        self.coeff_small_preds_loss = coeff_small_preds_loss
        self.weights = weights

    def forward(
        self,
        logits : torch.Tensor,
        labels: torch.Tensor,
        preds: torch.Tensor,
        batch_df_odds: torch.Tensor,
        amount_base=10
    ):
        """Calculate loss with optional profit component
        
        Args:
            logits: Model outputs
            labels: True labels
            batch_df: DataFrame containing original odds/features for this batch
        """
        # Standard cross entropy
        if self.weights is not None:
            ce_loss = F.cross_entropy(logits, labels, weight=self.weights)
        else:
            ce_loss = F.cross_entropy(logits, labels)

        ce_loss *= self.coeff_ce_loss
        
        # Add profit component
        probs = torch.softmax(logits, dim=1)[:, 1]  # Class 1 probabilities
        profit_loss = func_profit_loss(
            pred_probs=probs,
            labels=labels,
            batch_df_odds=batch_df_odds,
            amount_base=amount_base
        )

        profit_loss *= self.coeff_profit_loss

        # Penalize when the number of predictions is too different from the number of labels
        # This is to avoid models that predict too few positive cases and also to avoid models that predict too many positive cases
        pred_sum = torch.sum(preds)
        label_sum = torch.sum(labels)

        # Avoid division by zero, add epsilon
        eps = 1e-6
        ratio = pred_sum / (label_sum + eps)
        small_preds_loss = (1.0 - ratio).pow(2)

        small_preds_loss *= self.coeff_small_preds_loss

        return ce_loss + profit_loss + small_preds_loss