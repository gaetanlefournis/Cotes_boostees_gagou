import argparse

from ai_model.Evolutionary_algorithm.evolutionary_optimizer import \
    EvolutionaryOptimizer
from ai_model.Evolutionary_algorithm.evolutionary_visualizer import \
    EvolutionaryVisualizer
from ai_model.main_ai_model import MainTrainingAIModel
from ai_model.prepare_data.prepare_data import EnhancedPrepareData
from utils.constants import LIST_SEEDS
from utils.tools import load_config
from utils.tools_ai_model import load_pickle_file, save_pickle_file

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
        target_fitness = 821

        for seed in list_seeds:
            config['PREPARE_DATA']['random_state'] = seed
            print(f"Running with seed: {seed}")
            config['MLFLOW']['run_name'] = f"Finding target fitness with seed {seed}"
            main_training_ai_model = MainTrainingAIModel(config)

            # Prepare the data
            print("Preparing the data...")
            prepare_data_instance = EnhancedPrepareData(df=main_training_ai_model.data, **config['PREPARE_DATA'])

            dictionary_file_name = {**config["PREPARE_DATA"]}

            data_object = load_pickle_file(**dictionary_file_name)
            if data_object is None:
                print(f"The Data file does not exist, running the preparation again.")
                # Prepare the data
                print("Preparing the data...")
                prepare_data_instance()
                # Save the prepared data to a pickle file
                save_pickle_file(prepare_data_instance, **dictionary_file_name)

                data_object = prepare_data_instance
            else:
                print("Data file loaded")

            # Get the training, validation, and test sets
            X_train, y_train, X_val, y_val, X_test, y_test = data_object.X_train, data_object.y_train, data_object.X_val, data_object.y_val, data_object.X_test, data_object.y_test

            try:
                _, _, total_amount_naive_golden = main_training_ai_model.run(
                    (X_train, y_train), (X_val, y_val), (X_test, y_test), train_df=data_object.final_train_df, val_df=data_object.final_val_df, test_df=data_object.final_test_df
                )

                # Log parameters
                data_object.log_parameters(mlflow=main_training_ai_model.mlflow)
            except ValueError:
                continue
            finally:
                # Clean up resources
                main_training_ai_model.close()

            total_amount_naive_golden +=  total_amount_naive_golden

        # We define the target fitness as 3 times more than the average amount won with the naive strategy on gold odds
        target_fitness = total_amount_naive_golden / len(list_seeds) * 3 if total_amount_naive_golden != 0 else target_fitness * 3
        print(f"Target fitness for early stopping: {target_fitness:.2f}")

        # Run the evolutionary optimization process
        # In your main script, modify the evolutionary_optimizer.main() call:
        evolutionary_optimizer.main(
            base_config=config,
            target_fitness=target_fitness,
            checkpoint_path="checkpoints/checkpoint_gen_8.pkl"
        )

        # Write the updated index back to the file
        with open("index.txt", "w") as f:
            f.write(str(index))
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


    # python3 ai_model/main_ea.py --config_path config/config.yaml --env_path config/.env.gagou --hyperparameters_path config/hyperparameters_range.yaml
    # python3 -m ai_model.main_ea --config_path config/config.yaml --env_path config/.env.gagou --hyperparameters_path config/hyperparameters_range.yaml

    # mlflow ui --backend-store-uri file:///home/gagou/Documents/Projet/Cotes_boostees_gagou/mlruns --host 0.0.0.0 --port 5002