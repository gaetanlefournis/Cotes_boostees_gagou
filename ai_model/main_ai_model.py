import argparse
import os
import time

import mlflow
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from ai_model.mlflow_perso.mlflow_perso import MLFlow
from ai_model.models.tester import Tester
from ai_model.models.trainer import Trainer
from ai_model.prepare_data.prepare_data import EnhancedPrepareData
from boosted_odds.database.main_database import Database
from utils.constants import LIST_MODELS
from utils.tools import load_config
from utils.tools_ai_model import (load_pickle_file, save_pickle_file,
                                  save_results)

LIST_WEBSITES = ["winamax", "PSEL", "betclic", "unibet"] # "winamax", "PSEL", "betclic", "unibet"

class MainTrainingAIModel():
    """
    Main class to process the training of AI models.
    
    The program will train several algorithms for each site in the config file.
    
    Args:
        config_path (str): Path to the configuration file.
        env_path (str): Path to the environment file.
        website (str): The website for which the model is being trained. Default is "winamax".
    """
    def __init__(
        self,
        config: dict,
        website: str = "winamax"
    ) -> None:
        self.config = config
        self.database = None
        self.mlflow = None
        self.model_type = None
        self.model = None
        self.device = None
        self.config_number = None
        self.data = None

        self.amount_won_model = 0
        self.list_amount_won = []

        self._initialize_all(website=website)
        

    def _initialize_all(self, website: str) -> None:
        """
        Initialize the training process.
        This includes setting up the database connection and MLflow tracking.
        """
        self.database = Database(**self.config['DB_VPS'])
        self.mlflow = MLFlow(**self.config['MLFLOW'])
        self.config_number = self.config['config_number']
        self.model_type = LIST_MODELS[self.config['AI_MODEL_CONFIGS']['config_' + str(self.config_number)]['model_name']]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = self.database.retrieve_all(table=website)

    def _create_data_loader(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 64,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Create data loaders for training and testing.
        
        Args:
            X (torch.Tensor): Features tensor.
            y (torch.Tensor): Labels tensor.
            batch_size (int): Size of each batch.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: DataLoader instance for the dataset.
        """
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def _create_data_loaders(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        X_val: torch.Tensor = None,
        y_val: torch.Tensor = None
    ) -> tuple:
        """
        Create data loaders for training, testing, and validation datasets.
        
        Args:
            X_train (torch.Tensor): Training features tensor.
            y_train (torch.Tensor): Training labels tensor.
            X_test (torch.Tensor): Testing features tensor.
            y_test (torch.Tensor): Testing labels tensor.
            X_val (torch.Tensor, optional): Validation features tensor. Defaults to None.
            y_val (torch.Tensor, optional): Validation labels tensor. Defaults to None.
        
        Returns:
            tuple: DataLoader instances for training, testing, and validation datasets.
        """
        # Prepare data loaders for training and validation
        if isinstance(self.model, torch.nn.Module):
            train_loader = self._create_data_loader(X_train, y_train, batch_size=self.config['TRAINING']['batch_size'], shuffle=False)
            val_loader = self._create_data_loader(X_val, y_val, batch_size=self.config['TRAINING']['batch_size'], shuffle=False) if X_val is not None else None
            test_loader = self._create_data_loader(X_test, y_test, batch_size=self.config['TESTING']['test_batch_size'], shuffle=False)
        else:
            train_loader = (X_train, y_train)
            test_loader = (X_test, y_test)
            val_loader = (X_val, y_val) if X_val is not None else None

        return train_loader, test_loader, val_loader
    
    def _create_dataframe_loaders(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        val_df: pd.DataFrame = None
    ) -> tuple:
        """
        Create data loaders for original untransformed DataFrames that correspond to feature-transformed loaders.
        Maintains same order/batches as transformed data.
        
        Args:
            train_df (pd.DataFrame): Untransformed training DataFrame
            test_df (pd.DataFrame): Untransformed test DataFrame
            val_df (pd.DataFrame, optional): Untransformed validation DataFrame
        
        Returns:
            tuple: DataLoader instances for original DataFrames
        """
        def create_df_loader(df, batch_size):
            if df is None or len(df) == 0:
                return None
                
            # Convert DataFrame to tensor dataset
            df_tensor = torch.utils.data.TensorDataset(
                torch.arange(len(df)),  # Using indices as dummy input
                torch.from_numpy(df['odd'].values.astype(float)),
                torch.from_numpy(df['result'].map({'Gagné': 1, 'Perdu': 0}).values)
            )
            
            return torch.utils.data.DataLoader(
                df_tensor,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False
            )

        train_loader = create_df_loader(train_df, self.config['TRAINING']['batch_size'])
        test_loader = create_df_loader(test_df, self.config['TESTING']['test_batch_size'])
        val_loader = create_df_loader(val_df, self.config['TRAINING']['batch_size']) if val_df is not None else None

        return train_loader, test_loader, val_loader

    def run(self, data_train: tuple[torch.Tensor], data_val: tuple[torch.Tensor], data_test: tuple[torch.Tensor], train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[list, float, float] | int:
        """
        Run the training process for the AI model.
        """
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["MKL_NUM_THREADS"] = "2"

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # End the run if one is active
        if mlflow.active_run():
            mlflow.end_run()

        # Start MLflow run
        self.mlflow.start_run()

        if self.mlflow.already_exists:
            print("Existing run.")
            raise ValueError("run already exists")

        # Prepare the data - now returns (X_train, y_train), (X_val, y_val), (X_test, y_test)
        X_train, y_train, X_val, y_val, X_test, y_test = data_train[0], data_train[1], data_val[0], data_val[1], data_test[0], data_test[1]
        print("\n\nData preparation completed.")

        # count the number of targets equal to 0 and equal to 1 in y_train and y_test and weights the classes during training
        num_zeros_train = (y_train == 0).sum().item()
        num_ones_train = (y_train == 1).sum().item()
        weights = torch.tensor([num_ones_train / (num_zeros_train + num_ones_train), num_zeros_train / (num_zeros_train + num_ones_train)], device=self.device)

        # Initialize the model
        self.model = self.model_type(input_dim=X_train.shape[1])
        self.model.to(self.device)

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            mlflow_client=self.mlflow,
            device=self.device,
            weights=weights,
            **self.config['TRAINING']
        )
        
        # Create data loaders, and create splits of the df that correspond to the data loaders
        train_loader, test_loader, val_loader = self._create_data_loaders(X_train, y_train, X_test, y_test, X_val, y_val)
        train_df_loader, test_df_loader, val_df_loader = self._create_dataframe_loaders(train_df, test_df, val_df)

        # Train the model
        print("\n\nStarting training...")
        trained_model = trainer.train(
            train_loader=train_loader,
            train_df_loader=train_df_loader,
            val_loader=val_loader,
            val_df_loader=val_df_loader
        )

        # Initialize tester
        tester = Tester(
            model=trained_model,
            mlflow_client=self.mlflow,
            device=self.device,
            **self.config['TESTING']
        )

        # Test the model
        print("\n\nStarting testing...")
        total_amount_list_model, total_amount_model, total_amount_golden_naive = tester.test(
            test_loader=test_loader,
            test_df_loader=test_df_loader,
            test_df=test_df,
            prefix="test"
        )
        
        # Save model
        if isinstance(trained_model, torch.nn.Module):
            torch.save(trained_model.state_dict(), "final_model.pt")
            self.mlflow.log_artifacts("final_model.pt")
        else:
            import joblib
            joblib.dump(trained_model, "final_model.joblib")
            self.mlflow.log_artifacts("final_model.joblib")

        return total_amount_list_model, total_amount_model, total_amount_golden_naive

    def close(self) -> None:
        """
        Close the database connection and MLflow run.
        """
        if self.database:
            self.database.close()
        if self.mlflow:
            self.mlflow.end_run()



###########################################
#         Main execution block            #
###########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script with config and env file paths."
    )
    parser.add_argument(
        "--config_path",
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--env_path",
        default=".env.example",
        help="Path to the environment file (default: .env.example)",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Load the initial configuration
    config = load_config(args.config_path, args.env_path)

    # Create the lists of parameters
    list_seeds = [40, 41, 42, 43, 44, 45]
    list_models = [1, 2, 3, 4]
    save_str = "final_test_4"
    dictionary = {}

    # Create a loop to train the models with different configurations
    for model in list_models:
        dictionary[model] = {}
        for seed in list_seeds:
            try:
                # Update the config with the current seed and model
                config['PREPARE_DATA']['random_state'] = seed
                config['config_number'] = model
                config['MLFLOW']['run_name'] = f"model {model}, Seed: {seed}"

                # Create an instance of MainTrainingAIModel
                print("try to create an instance of MainTrainingAIModel")
                main_training = MainTrainingAIModel(config=config)

                # Prepare the data
                prepare_data_instance = EnhancedPrepareData(df=main_training.data, **config['PREPARE_DATA'])

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

                X_train, y_train, X_val, y_val, X_test, y_test = data_object.X_train, data_object.y_train, data_object.X_val, data_object.y_val, data_object.X_test, data_object.y_test

                print("number of features:", X_train.shape[1])

                # Run training
                try: 
                    total_amount_list_model, total_amount_model, total_amount_naive_golden = main_training.run(
                        data_train=(X_train, y_train),
                        data_val=(X_val, y_val),
                        data_test=(X_test, y_test),
                        train_df=data_object.final_train_df,
                        val_df=data_object.final_val_df,
                        test_df=data_object.final_test_df
                    )

                    data_object.log_parameters(mlflow=main_training.mlflow)
                except ValueError:
                    continue

                # Store the results in the dictionary
                dictionary[model][seed] = {
                    'total_amount_won': total_amount_model,
                    'total_amount_list': total_amount_list_model,
                    'total_amount_golden_naive': total_amount_naive_golden
                }

                # Clean up resources
                main_training.close()
            except Exception as e:
                print(f"An error occurred: {e}")
                raise
            finally:
                if 'main_training' in locals():
                    main_training.close()
                    print("Training process completed and resources cleaned up.")
                time.sleep(5)

    # Plot the results of the dictionary and save it
    save_results(dictionary, save_str)

    
    # python3 ai_model/main_ai_model.py --config_path config/config.yaml --env_path config/.env.gagou
    # python3 -m ai_model.main_ai_model --config_path config/config.yaml --env_path config/.env.gagou

    # mlflow ui --backend-store-uri file:///home/gagou/Documents/Projet/Cotes_boostees_gagou/mlruns --host 0.0.0.0 --port 5002