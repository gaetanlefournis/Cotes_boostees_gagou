import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tools_ai_model import func_profit_loss


class ProfitAwareLoss(nn.Module):
    """Profit-aware loss function combining cross-entropy and profit loss"""
    def __init__(
        self,
        number_loss: int = 1,
        type_profit_loss: int = 1,
        coeff_ce_loss : float = 0.5,
        coeff_profit_loss : float = 0.5,
        weights : torch.Tensor = None
    ):
        super().__init__()
        self.number_loss = number_loss
        self.type_profit_loss = type_profit_loss
        self.coeff_ce_loss = coeff_ce_loss
        self.coeff_profit_loss = coeff_profit_loss
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
        
        # Add profit component
        probs = torch.softmax(logits, dim=1)[:, 1]
        profit_loss = func_profit_loss(
            pred_probs=probs,
            labels=labels,
            batch_df_odds=batch_df_odds,
            amount_base=amount_base,
            type_loss=self.type_profit_loss,
        )

        if self.number_loss == 1:
            total_loss = (
                self.coeff_ce_loss * ce_loss +
                self.coeff_profit_loss * profit_loss
            )
        elif self.number_loss == 2:
            total_loss = (
                ce_loss ** self.coeff_ce_loss +
                profit_loss ** self.coeff_profit_loss
            )
        elif self.number_loss == 3:
            total_loss = (
                ce_loss ** self.coeff_ce_loss *
                profit_loss ** self.coeff_profit_loss
            )

        return total_loss