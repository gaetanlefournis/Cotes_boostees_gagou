import matplotlib.pyplot as plt
import pandas as pd
import torch
from pandas.plotting import parallel_coordinates


def func_profit_loss(
    pred_probs: torch.Tensor, 
    labels: torch.Tensor, 
    batch_df_odds: torch.Tensor,
    amount_base: float = 10
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
    
    # Theoretical bounds for normalization
    max_profit = ((batch_odds - 1) * amount_base * wins).sum()
    min_profit = -(amount_base * (1 - wins)).sum()
    
    # Normalize to [0,1] range
    normalized_loss = 1 - (batch_profit - min_profit) / (max_profit - min_profit + 1e-6)
    normalized_loss = torch.clamp(normalized_loss, 0, 1)
    
    return normalized_loss

def func_small_preds_loss(
    pred_probs: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Small predictions loss function penalizing small prediction values.
    """
    # Penalize when the number of predictions is too different from the number of labels
    # This is to avoid models that predict too few positive cases and also to avoid models that predict too many positive cases
    pred_sum = torch.sum(pred_probs)  # Sum of probabilities (differentiable)
    label_sum = torch.sum(labels)
    eps = 1e-6
    ratio = pred_sum / (label_sum + eps)
    loss = (1.0 - ratio)**2

    return loss

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

def plot_fitness_progression(stats: dict, index: int = 0):
    """Plot the best, average, and worst fitness over generations."""
    plt.figure(figsize=(10, 6))
    plt.plot(stats['generations'], stats['best_fitness'], 'g-', label='Best Fitness')
    plt.plot(stats['generations'], stats['avg_fitness'], 'b-', label='Average Fitness')
    plt.plot(stats['generations'], stats['worst_fitness'], 'r-', label='Worst Fitness')
    
    plt.title('Fitness Progression Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/fitness_progression_{index}.png')
    plt.close()

def plot_parameter_distribution(stats: dict, param_name: str, index: int = 0):
    """Plot the distribution of a parameter's values across generations."""
    # Extract the parameter values from best individuals
    param_values = [ind[param_name]['value'] for ind in stats['best_individuals']]
    
    plt.figure(figsize=(10, 6))
    plt.plot(stats['generations'], param_values, 'bo-')
    
    plt.title(f'Evolution of {param_name} Parameter')
    plt.xlabel('Generation')
    plt.ylabel('Parameter Value')
    plt.grid(True)
    plt.savefig(f'plots/{param_name}_distribution_{index}.png')
    plt.close()

def plot_parallel_coordinates(stats: dict, param_names: list, index: int = 0):
    """Create a parallel coordinates plot for multiple parameters."""
    # Prepare data
    data = []
    for i, gen in enumerate(stats['generations']):
        entry = {'Generation': gen, 'Fitness': stats['best_fitness'][i]}
        for param in param_names:
            entry[param] = stats['best_individuals'][i][param]['value']
        data.append(entry)
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 8))
    parallel_coordinates(df, 'Generation', color=('r', 'g', 'b'))
    plt.title('Parameter Values Across Generations')
    plt.xticks(rotation=10)
    plt.grid(True)
    plt.savefig(f'plots/parallel_coordinates_{index}.png')
    plt.close()