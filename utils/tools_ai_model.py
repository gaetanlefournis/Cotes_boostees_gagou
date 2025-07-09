import os
import pickle

import matplotlib.pyplot as plt
import torch
import random
import numpy as np

from ai_model.prepare_data.prepare_data import EnhancedPrepareData


def func_profit_loss(
    pred_probs: torch.Tensor, 
    labels: torch.Tensor, 
    batch_df_odds: torch.Tensor,
    amount_base: float = 10,
    type_loss: int = 1,
) -> torch.Tensor:
    """
    Differentiable profit loss (net profit version)
    
    Args:
        pred_probs: Probability tensor [batch_size] of class 1 (win)
        labels: Ground truth labels [batch_size] (0=loss, 1=win)
        batch_df: DataFrame slice containing 'odd' column
        amount_base: Base bet amount
    
    Returns:
        Normalized profit loss where:
        - 0 = maximum possible profit
        - 1 = maximum possible loss
    """
    # Clamp probabilities for numerical stability
    pred_probs = torch.clamp(pred_probs, 1e-3, 1 - 1e-3)

    # Convert odds to tensor on correct device
    batch_odds = batch_df_odds.to(pred_probs.device)

    # Profit/loss components
    wins = (labels == 1).float()
    losses = (labels == 0).float()
    
    # Symmetric profit/loss scaling
    net_profit = (
        (pred_probs * (batch_odds - 1) * wins - 
        (1 - pred_probs) * losses)
    ) * amount_base
    
    batch_profit = net_profit.sum()
    
    if type_loss == 1:
        # use the sigmoid function to create a penalty term
        normalized_loss = torch.sigmoid(-batch_profit/100)

    else:
        # Theoretical bounds for normalization
        max_profit = ((batch_odds - 1) * amount_base * wins).sum()
        min_profit = -(amount_base * (1 - wins)).sum()
        
        # Normalize to [0,1] range
        normalized_loss = 1 - (batch_profit - min_profit) / (max_profit - min_profit + 1e-6)
        normalized_loss = torch.clamp(normalized_loss, 0, 1)
    
    return normalized_loss

def save_results(dictionary: dict, save_str: str) -> None:
    """plot and save the figure of the results"""
    _, ax = plt.subplots(len(dictionary)//2 + 1 if len(dictionary) % 2 == 1 else len(dictionary)//2, 2, figsize=(16, 6))
    ax = ax.flatten()
    # calculate the average amount won for each characteristic by averaging on the seeds
    for i, (characteristic, seeds) in enumerate(dictionary.items()):
        total_sum = 0
        list_of_lists = []
        ax[i].set_xlabel("Number of bets")
        ax[i].set_ylabel("Total amount won")
        ax[i].set_title(f"Results for {save_str}: {characteristic}")
        for seed, data in seeds.items():
            total_sum += data['total_amount_won']
            list_of_lists.append(data['total_amount_list'])
            x = list(range(len(data['total_amount_list'])))
            ax[i].plot(x, data['total_amount_list'], label=f"Seed: {seed}")
            ax[i].legend()

        avg_amount_won = total_sum / len(seeds)
        ax[i].hlines(avg_amount_won, xmin=0, xmax=len(list_of_lists[0])-1, label=f"{save_str}: {characteristic}, Avg amount won: {avg_amount_won:.2f}")
        ax[i].legend()
    plt.tight_layout()
    plt.savefig(f"plots/results_{save_str}.png")
    plt.close()

def save_pickle_file(object: object, **kwargs) -> None:
    """Save the object to a pickle file with a dynamic path."""
    path_name = "ai_model/Data/" + "_".join([str(k) + '_' + str(v) for k, v in kwargs.items()]) + ".pkl"
    # if the path already exists, we don't save
    if not os.path.exists(path_name):
        # Create the directory and file if it doesn't exist
        os.makedirs(os.path.dirname(path_name), exist_ok=True)
        with open(path_name, 'wb') as f:
            pickle.dump(object, f)
    else:
        print(f"File {path_name} already exists. Not overwriting.")

def load_pickle_file(**kwargs) -> 'EnhancedPrepareData':
    """Load the object from a pickle file."""
    path_name = "ai_model/Data/" + "_".join([str(k) + '_' + str(v) for k, v in kwargs.items()]) + ".pkl"
    # if the path already exists, we load
    if os.path.exists(path_name):
        with open(path_name, 'rb') as f:
            return pickle.load(f)

def load_checkpoint(checkpoint_path: str) -> tuple:
    """Load a saved checkpoint and return the optimization state."""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Restore random states
    random.setstate(checkpoint['random_state'])
    np.random.set_state(checkpoint['numpy_random_state'])

    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    return (
        checkpoint['generation'],
        checkpoint['population'],
        checkpoint['config_hyperparameters'],
        checkpoint['optimizer_params'],
        checkpoint['stats'],
    )