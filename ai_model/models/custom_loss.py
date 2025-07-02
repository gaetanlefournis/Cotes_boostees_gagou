import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tools_ai_model import func_profit_loss, func_small_preds_loss


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
        ce_loss = ce_loss ** self.coeff_ce_loss
        
        # Add profit component
        probs = torch.softmax(logits, dim=1)[:, 1]
        profit_loss = func_profit_loss(
            pred_probs=probs,
            labels=labels,
            batch_df_odds=batch_df_odds,
            amount_base=amount_base
        )
        profit_loss = profit_loss ** (1.0 + self.coeff_profit_loss)

        # Add penalty component
        small_preds_loss = func_small_preds_loss(
            pred_probs=probs,
            labels=labels
        )
        small_preds_loss = small_preds_loss ** (0.5 * self.coeff_small_preds_loss)

        return ce_loss * profit_loss * small_preds_loss