import os

from ai_model.Evolutionary_algorithm.evolutionary_visualizer import \
    EvolutionaryVisualizer
from utils.tools_ai_model import load_checkpoint
import argparse
from utils.tools import load_config


def main_visualizer():
    """
    Load the checkpoints one by one and store the results in a big dictionary stats.
    """
    final_stats = {
        'population': [],
        'generations': [],
        'best_fitness': [],
        'avg_fitness': [],
        'worst_fitness': [],
        'best_individuals': []
    }

    LIST_FILE = ['checkpoints/checkpoint_gen_1.pkl', 'checkpoints/checkpoint_gen_2.pkl', 'checkpoints/checkpoint_gen_3.pkl', 'checkpoints/checkpoint_gen_3.pkl', 'checkpoints/checkpoint_gen_4.pkl', 'checkpoints/checkpoint_gen_5.pkl', 'checkpoints/checkpoint_gen_6.pkl', 'checkpoints/checkpoint_gen_7.pkl', 'checkpoints_1/checkpoint_gen_8.pkl', 'checkpoints_1/checkpoint_gen_9.pkl', 'checkpoints_1/checkpoint_gen_10.pkl', 'checkpoints_1/checkpoint_gen_11.pkl', 'checkpoints_1/checkpoint_gen_12.pkl', 'checkpoints_1/checkpoint_gen_13.pkl', 'checkpoints_1/checkpoint_gen_14.pkl', 'checkpoints_1/checkpoint_gen_15.pkl', 'checkpoints_1/checkpoint_gen_16.pkl', 'checkpoints_1/checkpoint_gen_17.pkl', 'checkpoints_1/checkpoint_gen_18.pkl', 'checkpoints_1/checkpoint_gen_19.pkl', 'checkpoints_1/checkpoint_gen_20.pkl']
    for file in LIST_FILE:
        if file.endswith(".pkl"):
            # Load the checkpoint
            checkpoint_path = os.path.join("", file)
            print(f"Loading checkpoint: {checkpoint_path}")
            _, _, _, _, stats = load_checkpoint(checkpoint_path)

            # create the stats dictionary
            if type(stats['generations']) == list:
                final_stats['generations'].append(stats['generations'][-1])
                final_stats['best_fitness'].append(stats['best_fitness'][-1])
                final_stats['avg_fitness'].append(stats['avg_fitness'][-1])
                final_stats['worst_fitness'].append(stats['worst_fitness'][-1])
                final_stats['best_individuals'].append(stats['best_individuals'][-1])
                final_stats['population'].append(stats['population'][-1])
            elif type(stats['generations']) == int:
                final_stats['generations'].append(stats['generations'])
                final_stats['best_fitness'].append(stats['best_fitness'])
                final_stats['avg_fitness'].append(stats['avg_fitness'])
                final_stats['worst_fitness'].append(stats['worst_fitness'])
                final_stats['best_individuals'].append(stats['best_individuals'])
                final_stats['population'].append(stats['population'])

    print("best individual:", final_stats['best_individuals'][-1])
    print("with fitness:", final_stats['best_fitness'][-1])

    return final_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script with config and env file paths."
    )

    parser.add_argument(
        "--hyperparameters_path",
        default="hyperparameters_range.yaml",
        help="Path to the hyperparameters configuration file (default: config/hyperparameters_range.yaml)",
    )

    args = parser.parse_args()

    config_hyperparameters = load_config(args.hyperparameters_path)

    stats = main_visualizer()
    # Create an instance of EvolutionaryVisualizer
    visualizer = EvolutionaryVisualizer(stats, config_hyperparameters, output_dir="evolution_plots_test")
    
    # Plot the results
    visualizer.generate_all_plots()