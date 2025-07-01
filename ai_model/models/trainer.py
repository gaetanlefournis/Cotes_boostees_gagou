import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ai_model.mlflow.mlflow import MLFlow
from ai_model.models.custom_loss import ProfitAwareLoss


class Trainer:
    """    Trainer class for PyTorch models with profit-based loss function.
    This class handles training, validation, and logging of metrics using MLflow.
    It supports both profit-based loss and standard cross-entropy loss with optional class weights.
    Args:
        model (object): PyTorch model to be trained.
        mlflow_client (MLFlow): MLFlow client instance for logging.
        device (torch.device): Device to run the model on (CPU or GPU).
        weights (torch.Tensor, optional): Class weights for the loss function.
    """
    def __init__(
        self,
        model: object,
        mlflow_client: MLFlow,
        device: torch.device,
        weights: torch.Tensor = None,
        batch_size: int = 64,
        epochs: int = 10,
        learning_rate: float = 0.0001,
        patience: int = 20,
        coefficient_ce_loss: float = 0.25,
        coefficient_profit_loss: float = 0.25,
        coefficient_small_preds_loss: float = 0.5,
        **kwargs,
    ):
        self.model = model
        self.mlflow = mlflow_client
        self.device = device
        self.weights = weights
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.coefficient_ce_loss = coefficient_ce_loss
        self.coefficient_profit_loss = coefficient_profit_loss
        self.coefficient_small_preds_loss = coefficient_small_preds_loss

        # Send model to device
        self.model.to(self.device)

        # normal loss criterion (with weights if provided)
        self.loss_criterion = None
        self.loss_criterion_validation = None
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _log_loss_accuracy(
        self,
        train_loss: float,
        val_loss: float,
        train_accuracy: float,
        val_accuracy: float
    ) -> None:
        """Log loss and accuracy plots"""
        # Log the graphs for training and validation accuracy and loss
        plt.figure(figsize=(6, 3))
        plt.plot(train_accuracy, label='Train Accuracy')
        if val_accuracy:
            plt.plot(val_accuracy, label='Val Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.grid(True, alpha=0.2)
        acc_path = "accuracy_plot.png"
        plt.savefig(acc_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.mlflow.log_artifacts(acc_path)
        os.remove(acc_path)

        plt.figure(figsize=(6, 3))
        plt.plot(train_loss, label='Train Loss')
        if val_loss:
            plt.plot(val_loss, label='Val Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.2)
        loss_path = "loss_plot.png"
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.mlflow.log_artifacts(loss_path)
        os.remove(loss_path)

    def train(
        self,
        train_loader: DataLoader,
        train_df_loader: DataLoader = None,
        val_loader: DataLoader = None,
        val_df_loader: DataLoader = None,
    ):
        """Main training method. Use early stopping if validation loader is provided."""
        # Initialize the metrics outside of epoch loop
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        # Define the 2 loss functions
        self.loss_criterion = ProfitAwareLoss(
            coeff_ce_loss=self.coefficient_ce_loss,
            coeff_profit_loss=self.coefficient_profit_loss,
            coeff_small_preds_loss=self.coefficient_small_preds_loss,
            weights=self.weights
        ).to(self.device)
        self.loss_criterion_validation = ProfitAwareLoss(
            coeff_ce_loss=self.coefficient_ce_loss,
            coeff_profit_loss=self.coefficient_profit_loss,
            coeff_small_preds_loss=self.coefficient_small_preds_loss,
            weights=None  # No weights for validation loss
        ).to(self.device)

        # Initialize early stopping variables (patience = self.patience)
        best_model_state = copy.deepcopy(self.model.state_dict())
        best_val_loss = np.inf
        patience_counter = 0

        for epoch in tqdm(range(self.epochs), desc="Training Epochs"):

            self.model.train()
            train_loss = 0
            train_correct = 0

            if train_df_loader:
                combined_iter = zip(train_loader, train_df_loader)
            else:
                combined_iter = ((batch, None) for batch in train_loader)

            for (inputs, labels), df_data in combined_iter:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                batch_df_odds = df_data[1] if df_data else None
                
                # Do the predictions
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                # Use our own loss function
                loss = self.loss_criterion(outputs, labels, preds, batch_df_odds=batch_df_odds)

                # Do the backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                train_loss += loss.item() * inputs.size(0)  
                train_correct += (np.array(preds.cpu()) == np.array(labels.cpu())).sum().item()

            # Calculate average loss and accuracy for the epoch
            train_loss /= len(train_loader.dataset)
            train_accuracy = train_correct / len(train_loader.dataset)

            # Validation if val_loader is provided
            if val_loader is not None:

                self.model.eval()
                val_loss = 0
                val_correct = 0

                with torch.no_grad():

                    if val_df_loader:
                        val_combined_iter = zip(val_loader, val_df_loader)
                    else:
                        val_combined_iter = ((batch, None) for batch in val_loader)

                    for (inputs_val, labels_val), val_df_data in val_combined_iter:
                        val_inputs, val_labels = inputs_val.to(self.device), labels_val.to(self.device)
                        val_batch_df_odds = val_df_data[1] if val_df_data else None

                        # Do the predictions
                        val_outputs = self.model(val_inputs)
                        _, val_preds = torch.max(val_outputs, 1)

                        # Use our own loss function
                        loss = self.loss_criterion_validation(val_outputs, val_labels, val_preds, batch_df_odds=val_batch_df_odds)

                        # Update metrics
                        val_loss += loss.item() * val_inputs.size(0)
                        val_correct += (np.array(val_preds.cpu()) == np.array(val_labels.cpu())).sum().item()

                # Calculate average validation loss and accuracy for this epoch
                val_loss /= len(val_loader.dataset)
                val_accuracy = val_correct / len(val_loader.dataset)

                # append metrics for logging
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accuracies.append(train_accuracy)
                val_accuracies.append(val_accuracy)
                        
                self.mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                }, step=epoch)

                # Early stopping logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"⏹️ Early stopping triggered at epoch {epoch+1}.")
                        break

            else:
                # If no validation loader, just log training metrics
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)
                self.mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                }, step=epoch)


        # log accuracy and loss plots
        self._log_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies)

        # Log training parameters to MLflow
        self.log_training_parameters()

        # Save model to MLflow
        print("✅ Loading best model weights from early stopping...")
        self.model.load_state_dict(best_model_state)
        self.mlflow.log_model(self.model, str(self.model.__class__.__name__))
        print(f"✅ Model saved to MLflow with run_id: {self.mlflow.run_id}")
        
        return self.model

    def log_training_parameters(self):
        """Log training parameters to MLflow"""
        self.mlflow.log_params({
            "model_name": self.model.__class__.__name__,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "patience": self.patience,
            "coefficient_ce_loss": self.coefficient_ce_loss,
            "coefficient_profit_loss": self.coefficient_profit_loss,
            "coefficient_small_preds_loss": self.coefficient_small_preds_loss,
            "weights": self.weights.tolist() if self.weights is not None else None
        })