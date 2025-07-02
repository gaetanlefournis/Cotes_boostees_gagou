import argparse

from ai_model.Evolutionary_algorithm.evolutionary_optimizer import \
    EvolutionaryOptimizer
from ai_model.main_ai_model import MainTrainingAIModel
from utils.constants import LIST_SEEDS
from utils.tools import load_config
from utils.tools_ai_model import (plot_fitness_progression,
                                  plot_parallel_coordinates,
                                  plot_parameter_distribution)

LIST_WEBSITES = ["winamax", "PSEL", "betclic", "unibet"] # "winamax", "PSEL", "betclic", "unibet"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script with config and env file paths."
    )
    parser.add_argument(
        "--config_path",
        default="config.yaml",
        help="Path to the configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--env_path",
        default=".env.example",
        help="Path to the environment file (default: config/.env.example)",
    )
    parser.add_argument(
        "--hyperparameters_path",
        default="hyperparameters_range.yaml",
        help="Path to the hyperparameters configuration file (default: config/hyperparameters_range.yaml)",
    )

    

    try:
        # Read the index from the file "index.txt" and increment it
        with open("index.txt", "r") as f:
            index = int(f.read().strip())
        index += 1
        print(f"Current index: {index}")

        # Parse the arguments
        args = parser.parse_args()

        # Load the initial configuration
        config = load_config(args.config_path, args.env_path)
        config_hyperparameters = load_config(args.hyperparameters_path)

        # Create the instance of EvolutionaryOptimizer
        print("Creating an instance of EvolutionaryOptimizer")
        evolutionary_optimizer = EvolutionaryOptimizer(config_hyperparameters, **config['EVOLUTIONARY_ALGORITHM'])

        # Find the target fitness that will be used for early stopping
        list_seeds = LIST_SEEDS
        total_amount_naive_golden = 0
        target_fitness = 410

        for seed in list_seeds:
            config['SEED'] = seed
            print(f"Running with seed: {seed}")
            config['MLFLOW']['run_name'] = f"Finding target fitness with seed {seed}"
            main_training_ai_model = MainTrainingAIModel(config)
            try:
                _, _, total_amount_naive_golden = main_training_ai_model.run()
            except ValueError:
                continue
            finally:
                # Clean up resources
                main_training_ai_model.close()

            total_amount_naive_golden +=  total_amount_naive_golden

        # We define the target fitness as 50% more than the average amount won with the naive strategy on gold odds
        target_fitness = total_amount_naive_golden / len(list_seeds) * 1.5 if total_amount_naive_golden != 0 else target_fitness
        print(f"Target fitness for early stopping: {target_fitness:.2f}")

        # Run the evolutionary optimization process
        # In your main script, modify the evolutionary_optimizer.main() call:
        best_individual, stats = evolutionary_optimizer.main(
            base_config=config,
            target_fitness=target_fitness,
            checkpoint_interval=1,
            checkpoint_path="checkpoints/checkpoint_gen_6.pkl"
        )

        plot_fitness_progression(stats, index = index)
        for _, data in config_hyperparameters.items():
            for param_name in data.keys():
                plot_parameter_distribution(stats, param_name, index = index)
        plot_parallel_coordinates(stats, list(key for data in config_hyperparameters.values() for key in data.keys()), index = index)

        # Write the updated index back to the file
        with open("index.txt", "w") as f:
            f.write(str(index))
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


    # python3 ai_model/main_ea.py --config_path config/config.yaml --env_path config/.env.gagou --hyperparameters_path config/hyperparameters_range.yaml
    # python3 -m ai_model.main_ea --config_path config/config.yaml --env_path config/.env.gagou --hyperparameters_path config/hyperparameters_range.yaml

    # mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5001