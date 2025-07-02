import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ai_model.mlflow.mlflow import MLFlow
from ai_model.models.custom_loss import ProfitAwareLoss
from utils.constants import AMOUNT_BASE


class Tester:
    """Tester class for evaluating PyTorch models with enhanced type checking and logging."""
    def __init__(
        self,
        model: object,
        mlflow_client: MLFlow,
        device: torch.device,
        test_batch_size: int,
        test_coefficient_ce_loss: float,
        test_coefficient_profit_loss: float,
        test_coefficient_small_preds_loss: float,
        test_visualization_best_worst: bool,
        **kwargs,
    ):
        self.model = model
        self.mlflow = mlflow_client
        self.device = device
        self.test_batch_size = test_batch_size
        self.test_coefficient_ce_loss = test_coefficient_ce_loss
        self.test_coefficient_profit_loss = test_coefficient_profit_loss
        self.test_coefficient_small_preds_loss = test_coefficient_small_preds_loss
        self.test_visualization_best_worst = test_visualization_best_worst
        self.model.to(self.device)

        self.criterion_test = None

    def _log_confusion_matrix(
        self,
        y_true : np.ndarray,
        y_pred: np.ndarray,
        prefix: str
    ) -> None:
        """Enhanced confusion matrix with percentage view"""
        cm = confusion_matrix(y_true, y_pred)
        _, ax = plt.subplots(1, 2, figsize=(16, 6))
        
        # Absolute values
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
        ax[0].set_title(f"{prefix.capitalize()} Confusion Matrix (Counts)")
        ax[0].set_xlabel('Predicted')
        ax[0].set_ylabel('True Label')
        
        # Normalized view
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax[1])
        ax[1].set_title(f"{prefix.capitalize()} Confusion Matrix (Normalized)")
        ax[1].set_xlabel('Predicted')
        ax[1].set_ylabel('True Label')
        
        plt.tight_layout()
        cm_path = f"{prefix}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        self.mlflow.log_artifacts(cm_path)
        os.remove(cm_path)

    def _log_roc_curve(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        prefix: str
    ) -> None:
        """Enhanced ROC curve with better formatting"""
        
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'{prefix.capitalize()} ROC Curve', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        roc_path = f"{prefix}_roc_curve.png"
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.mlflow.log_artifacts(roc_path)
        os.remove(roc_path)        

    def test(
        self,
        test_loader: DataLoader,
        test_df_loader: DataLoader,  # Changed from pd.DataFrame to DataLoader
        test_df: pd.DataFrame,
        prefix: str = "test"
    ) -> tuple[list, float]:
        """Enhanced testing method with proper DataFrame batch handling"""
        
        self.model.eval()
        self.criterion_test = ProfitAwareLoss(
            coeff_ce_loss=self.test_coefficient_ce_loss,
            coeff_profit_loss=self.test_coefficient_profit_loss,
            coeff_small_preds_loss=self.test_coefficient_small_preds_loss,
            weights=None
        ).to(self.device)

        test_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        all_indices = []  # To track original DataFrame positions
        
        # Create combined iterator
        if test_df_loader:
            test_combined_iter = zip(test_loader, test_df_loader)
        else:
            test_combined_iter = ((batch, None) for batch in test_loader)

        with torch.no_grad():
            for (inputs, labels), df_batch in tqdm(test_combined_iter):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                batch_df_odds = df_batch[1] if df_batch is not None else None

                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                loss = self.criterion_test(outputs, labels, batch_df_odds=batch_df_odds)
                test_loss += loss.item()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_indices.extend(df_batch[0])

        # Print how many golden samples are in the test set
        num_golden_samples = test_df[test_df['golden'] == 'gold'].shape[0]

        # Plot the confusion matrix only for golden samples
        if num_golden_samples > 0:
            # Get indices of golden samples by resetting the index of the dataframe
            test_df_golden = test_df.copy()
            test_df_golden.reset_index(drop=True, inplace=True)
            golden_indices = test_df_golden[test_df_golden['golden'] == 'gold'].index
            golden_labels = [all_labels[i] for i in golden_indices]
            golden_preds = [all_preds[i] for i in golden_indices]
            golden_probs = [all_probs[i] for i in golden_indices]

            self._log_confusion_matrix(golden_labels, golden_preds, f"{prefix}_golden")
            self._log_roc_curve(golden_labels, golden_probs, f"{prefix}_golden")

        # Calculate total amount won based on model predictions
        total_amount_naive, total_amount_model, total_amount_best, total_amount_worst, total_amount_list_naive, total_amount_list_model, total_amount_list_best, total_amount_list_worst = self._find_total_amount_won(
            preds=torch.tensor(all_preds),
            labels=torch.tensor(all_labels),
            test_df=test_df,
            golden=False,
        )

        total_amount_naive_golden, total_amount_model_golden, total_amount_best_golden, total_amount_worst_golden, total_amount_list_naive_golden, total_amount_list_model_golden, total_amount_list_best_golden, total_amount_list_worst_golden = self._find_total_amount_won(
            preds=torch.tensor(all_preds),
            labels=torch.tensor(all_labels),
            test_df=test_df,
            golden=True,
        )

        # Plot the total amount won over time
        self._log_amount_won(
            total_amount_naive=total_amount_naive,
            total_amount_model=total_amount_model,
            total_amount_best=total_amount_best,
            total_amount_worst=total_amount_worst,
            total_amount_list_naive=total_amount_list_naive,
            total_amount_list_model=total_amount_list_model,
            total_amount_list_best=total_amount_list_best,
            total_amount_list_worst=total_amount_list_worst,
            prefix=prefix
        )

        # Plot the total amount won over time for golden predictions
        self._log_amount_won(
            total_amount_naive=total_amount_naive_golden,
            total_amount_model=total_amount_model_golden,
            total_amount_best=total_amount_best_golden,
            total_amount_worst=total_amount_worst_golden,
            total_amount_list_naive=total_amount_list_naive_golden,
            total_amount_list_model=total_amount_list_model_golden,
            total_amount_list_best=total_amount_list_best_golden,
            total_amount_list_worst=total_amount_list_worst_golden,
            prefix=f"{prefix}_golden"
        )


        # Safely calculate metrics
        metrics = {
            f"{prefix}_loss": test_loss / len(test_loader.dataset),  # Average per sample
            f"{prefix}_accuracy": accuracy_score(all_labels, all_preds),
            f"{prefix}_roc_auc": roc_auc_score(all_labels, all_probs)
        }
        
        # Add classification metrics if binary
        if len(np.unique(all_labels)) == 2:
            metrics.update({
                f"{prefix}_precision": precision_score(all_labels, all_preds),
                f"{prefix}_recall": recall_score(all_labels, all_preds),
                f"{prefix}_f1": f1_score(all_labels, all_preds)
            })
        
        # Logging
        self.mlflow.log_metrics(metrics)
        self.log_testing_parameters()
        self._log_confusion_matrix(all_labels, all_preds, prefix)
        self._log_roc_curve(all_labels, all_probs, prefix)
        
        return total_amount_list_model, total_amount_model, total_amount_naive_golden
    
    def _find_total_amount_won(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        test_df: pd.DataFrame,
        golden: bool = False,
    ) -> tuple[float, float, list, list]:
        """
        Calculate the total amount won based on model predictions.
        This is a helper function to be used in func_amount_won.
        """
        # A naive model will bet on everything, the other model will bet when it predicts 1
        total_amount_best = 0
        total_amount_naive = 0
        total_amount_model = 0
        total_amount_worst = 0
        total_amount_list_worst = [0]
        total_amount_list_best = [0]
        total_amount_list_naive = [0]
        total_amount_list_model = [0]

        # Calculate the amount won, thanks to the predictions and the column "odd" of the original test_df
        for i, pred, label in zip(range(len(preds)), preds, labels):
            if label == 1:
                if golden:
                    amount_won_naive = (test_df.iloc[i]['odd'] - 1) * AMOUNT_BASE if test_df.iloc[i]['golden'] == 'gold' else 0
                    amount_won_best = (test_df.iloc[i]['odd'] - 1) * AMOUNT_BASE if test_df.iloc[i]['golden'] == 'gold' else 0
                    amount_won_worst = 0
                else:
                    amount_won_naive = (test_df.iloc[i]['odd'] - 1) * AMOUNT_BASE
                    amount_won_best = (test_df.iloc[i]['odd'] - 1) * AMOUNT_BASE
                    amount_won_worst = 0
            else:
                if golden:
                    amount_won_naive = -AMOUNT_BASE if test_df.iloc[i]['golden'] == 'gold' else 0
                    amount_won_best = 0
                    amount_won_worst = -AMOUNT_BASE if test_df.iloc[i]['golden'] == 'gold' else 0
                else:
                    amount_won_naive = -AMOUNT_BASE
                    amount_won_best = 0
                    amount_won_worst = -AMOUNT_BASE

            if pred == 1:
                if golden:
                    if label == 1:
                        amount_won_model = (test_df.iloc[i]['odd'] - 1) * AMOUNT_BASE if test_df.iloc[i]['golden'] == 'gold' else 0
                    else:
                        amount_won_model = -AMOUNT_BASE if test_df.iloc[i]['golden'] == 'gold' else 0
                else:
                    if label == 1:
                        amount_won_model = (test_df.iloc[i]['odd'] - 1) * AMOUNT_BASE
                    else:
                        amount_won_model = -AMOUNT_BASE
            else:
                amount_won_model = 0
            total_amount_naive += amount_won_naive
            total_amount_model += amount_won_model
            total_amount_best += amount_won_best
            total_amount_worst += amount_won_worst
            total_amount_list_naive.append(total_amount_naive)
            total_amount_list_model.append(total_amount_model)
            total_amount_list_best.append(total_amount_best)
            total_amount_list_worst.append(total_amount_worst)

        return total_amount_naive, total_amount_model, total_amount_best, total_amount_worst, total_amount_list_naive, total_amount_list_model, total_amount_list_best, total_amount_list_worst

    def _log_amount_won(
        self,
        total_amount_naive: float,
        total_amount_model: float,
        total_amount_best: float,
        total_amount_worst: float,
        total_amount_list_naive: list,
        total_amount_list_model: list,
        total_amount_list_best: list,
        total_amount_list_worst: list,
        prefix: str = "test"
    ) -> float:
        """
        Calculate the total amount won based on model predictions.
        """

        # save as metric the last number of each list amount won
        self.mlflow.log_metrics({
            f"{prefix}_total_amount_won_naive": total_amount_naive,
            f"{prefix}_total_amount_won_model": total_amount_model,
            f"{prefix}_total_amount_won_best": total_amount_best,
            f"{prefix}_total_amount_won_worst": total_amount_worst
        })

        # Prepare the total amount lists for logging
        total_amount_lists = [
            (total_amount_list_naive, "naive", "g"),
            (total_amount_list_model, "model", "b"),
            (total_amount_list_best, "best", "m"),
            (total_amount_list_worst, "worst", "c")
        ] if self.test_visualization_best_worst else [
            (total_amount_list_naive, "naive", "g"),
            (total_amount_list_model, "model", "b")
        ]

        # Plot the total amount won over time
        plt.figure(figsize=(10, 6))
        for total_amount_list, label, color in total_amount_lists:
            plt.plot(total_amount_list, label=label, color=color)
        plt.title(f'Total Amount Won Over Time ({prefix.capitalize()})')
        plt.xlabel('Sample Index')
        plt.ylabel('Total Amount Won (€)')
        plt.axhline(0, color='red', linestyle='--', label='Break-even Point')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        total_amount_path = f"{prefix}_total_amount_won.png"
        plt.savefig(total_amount_path)
        plt.close()
        self.mlflow.log_artifacts(total_amount_path)
        os.remove(total_amount_path)

    def log_testing_parameters(self) -> None:
        """Log testing parameters to MLflow"""
        self.mlflow.log_params({
            "model_name": self.model.__class__.__name__,
            "test_batch_size": self.test_batch_size,
            "test_coefficient_ce_loss": self.test_coefficient_ce_loss,
            "test_coefficient_profit_loss": self.test_coefficient_profit_loss,
            "test_coefficient_small_preds_loss": self.test_coefficient_small_preds_loss,
            "test_visualization_best_worst": self.test_visualization_best_worst
        })