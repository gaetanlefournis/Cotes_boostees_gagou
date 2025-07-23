import copy
import os
import pickle
import random
from datetime import datetime

import numpy as np
from deap import base, creator, tools
from tqdm import tqdm

from ai_model.main_ai_model import MainTrainingAIModel
from ai_model.prepare_data.prepare_data import EnhancedPrepareData
from utils.constants import LIST_SEEDS
from utils.tools_ai_model import (load_checkpoint, load_pickle_file,
                                  save_pickle_file)


class EvolutionaryOptimizer():
    """    
    EvolutionaryOptimizer class for hyperparameter tuning using DEAP.
    This class implements an evolutionary algorithm to optimize hyperparameters
    for machine learning models. It supports various parameter types including
    categorical, uniform, and log-uniform distributions.
    """
    def __init__(
        self,
        config_hyperparameters : dict,
        population_size: int = 10,
        generations: int = 15,
        pc: float = 0.5,
        pm: float = 0.2,
        mutation_std_dev: float = 0.1,
        patience: int = 10,
        proper_initialize: bool = True,
        perform_crossover: bool = True,
        **kwargs,
    ) -> None:
        self.config_hyperparameters = config_hyperparameters
        self.population_size = population_size
        self.generations = generations
        self.pc = pc
        self.pm = pm
        self.mutation_std_dev = mutation_std_dev
        self.patience = patience
        self.proper_initialize = proper_initialize
        self.perform_crossover = perform_crossover
        self.setup_toolbox()

    def setup_toolbox(self):
        """Set up the DEAP toolbox with custom operators and individual creation."""
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", dict, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("generate_individual", self.generate_individual)
        self.toolbox.register("generate_population", tools.initRepeat, list, self.toolbox.generate_individual)
        self.toolbox.register("crossover", self.custom_crossover)
        self.toolbox.register("mutation", self.custom_mutate)
        self.toolbox.register("selection_elite", self.custom_selection_elite)
        self.toolbox.register("selection", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("clone", copy.deepcopy)

    def _assign_random_value(self, individual: dict, params: dict) -> dict:
        """Assign a random value to a parameter in the individual."""
        for param, values in params.items():
            if values['type'] == 'choice':
                individual[param] = {'value': random.choice(values['values']), 'type': 'choice', 'values': values['values']}
            elif values['type'] == 'uniform':
                individual[param] = {'value': random.uniform(values['low'], values['high']), 'type': 'uniform', 'low': values['low'], 'high': values['high']}
            elif values['type'] == 'loguniform':
                individual[param] = {'value': 10**random.uniform(np.log10(values['low']), np.log10(values['high'])), 'type': 'loguniform', 'low': values['low'], 'high': values['high']}
            else:
                raise ValueError(f"Unsupported parameter type: {values['type']} for parameter {param}")
        return individual

    def generate_individual(self, current_population: list[dict] = None) -> dict:
        """Generate a random individual based on the hyperparameter configuration.
        If 'proper_initialize' is True, it will ensure that the individual
        is initialized with values that are different from the current population values, otherwise it will generate a random individual."""
        individual = creator.Individual()
        
        # Training parameters
        for param_group in ['TRAINING_PARAMETERS', 'MODEL_PARAMETERS', 'DATA_PARAMETERS']:
            if param_group in self.config_hyperparameters:
                params = self.config_hyperparameters[param_group]
                if self.proper_initialize and current_population:
                    # Ensure the new individual has different values from the current population
                    existing_values_choice = [ind[param]['value'] for ind in current_population for param in params if ind[param]['type'] in ['choice']]
                    existing_values_uniform = [(ind[param]['value'] - 0.1 * (ind[param]['high'] - ind[param]['low']), ind[param]['value'] + 0.1 * (ind[param]['high'] - ind[param]['low'])) for ind in current_population for param in params if ind[param]['type'] in ['uniform']]
                    existing_values_loguniform = [(10**(ind[param]['value'] - 0.1 * (np.log10(ind[param]['high']) - np.log10(ind[param]['low']))), 10**(ind[param]['value'] + 0.1 * (np.log10(ind[param]['high']) - np.log10(ind[param]['low'])))) for ind in current_population for param in params if ind[param]['type'] in ['loguniform']]
                    individual = self._assign_random_value(individual, params)
                    
                    # If the assigned value already exists, reassign until it's unique
                    while any(individual[param]['value'] in existing_values_choice for param in params if individual[param]['type'] == 'choice') and any(existing[0] <= individual[param]['value'] <= existing[1] for existing in existing_values_uniform for param in params if individual[param]['type'] == 'uniform') and any(existing[0] <= individual[param]['value'] <= existing[1] for existing in existing_values_loguniform for param in params if individual[param]['type'] == 'loguniform'):
                        individual = self._assign_random_value(individual, params)
                else:
                    # Assign random values without checking for uniqueness
                    individual = self._assign_random_value(individual, params)

        # Ensure the output is DEAP-compatible
        if not isinstance(individual, dict):
            raise TypeError("Individual must be a dict")
        print(f"Generated individual: ")
        for k, v in individual.items():
            print(f"{k}: {v['value']}")
        return individual
    
    def generate_population(self) -> list[dict]:
        """Generate a population of individuals."""
        return self.toolbox.generate_population(self.population_size)

    def custom_crossover(self, individual1: dict, individual2: dict) -> tuple[dict, dict]:
        """Custom crossover operator that combines two individuals."""
        for param in individual1.keys():
            if random.random() < self.pc:
                # Swap values between two individuals for the given parameter
                individual1[param]['value'], individual2[param]['value'] = individual2[param]['value'], individual1[param]['value']

        return individual1, individual2

    def custom_mutate(self, individual: dict) -> tuple[dict]:
        """Custom mutation operator with local changes for numerical parameters."""
        print(f"intial individual before mutation: {[individual[param]['value'] for param in individual.keys()]}")
        for param, values in individual.items():
            if random.random() < self.pm:
                if values['type'] == 'choice':
                    # Pick a new value different from the current one
                    new_val = random.choice([v for v in values['values'] if v != values['value']])
                    individual[param]['value'] = new_val

                elif values['type'] == 'uniform':
                    current = values['value']
                    low = values['low']
                    high = values['high']
                    # Apply Gaussian noise centered around current value
                    std = self.mutation_std_dev * (high - low)
                    mutated = np.clip(np.random.normal(loc=current, scale=std), low, high)
                    individual[param]['value'] = mutated

                elif values['type'] == 'loguniform':
                    current = values['value']
                    low = values['low']
                    high = values['high']
                    # Work in log-space but same gaussian mutation logic
                    log_current = np.log10(current)
                    log_low = np.log10(low)
                    log_high = np.log10(high)
                    std = self.mutation_std_dev * (log_high - log_low)
                    mutated_log = np.clip(np.random.normal(loc=log_current, scale=std), log_low, log_high)
                    individual[param]['value'] = 10**mutated_log

        print(f"individual after mutation: {[individual[param]['value'] for param in individual.keys()]}")
        return individual,

    def custom_selection_elite(self, population: list[dict], n: int) -> list[dict]:
        """Custom selection operator that performs elitism selection, with n growing over time."""
        # Sort individuals by fitness
        sorted_population = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)
        
        # Select the top n individuals
        elite = tools.selBest(sorted_population, k=n)
        
        return elite

    def evaluate_individual(self, individual: dict, config: dict, indiv_number: int, gen_number: int) -> tuple[float]:
        """Evaluate the fitness of an individual."""
        # First, replace the configuration with the individual's parameters
        for param, value in individual.items():
            if param == 'config_number':
                config['config_number'] = value['value']
            elif param in config['TRAINING']:
                config['TRAINING'][param] = value['value']
                config['TESTING']['test_' + param] = value['value']  # Ensure testing config matches training
            elif param in config['PREPARE_DATA']:
                config['PREPARE_DATA'][param] = value['value']

        # Set a list of seeds to average the results over
        list_seeds = LIST_SEEDS
        total_amount_model_total = 0.0
        total_amount_model_list = [] # for plotting eventually but not used for now.

        for seed in list_seeds:
            config['PREPARE_DATA']['random_state'] = seed
            config['MLFLOW']['run_name'] = f"Run - Gen number {gen_number} - indiv number {indiv_number} with seed {seed}"
            config['MLFLOW']['experiment_name'] = f"Evolutionary Algorithm - Gen {gen_number}"

            # Create an instance of MainTrainingAIModel with the updated config
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

            # Get the training, validation, and test sets
            X_train, y_train, X_val, y_val, X_test, y_test = data_object.X_train, data_object.y_train, data_object.X_val, data_object.y_val, data_object.X_test, data_object.y_test

            # Run the training and get the total amount won
            try:
                _, total_amount_model, _ = main_training.run((X_train, y_train), (X_val, y_val), (X_test, y_test),
                                                             train_df=data_object.final_train_df, 
                                                             val_df=data_object.final_val_df, 
                                                             test_df=data_object.final_test_df)
                
                # Log parameters
                data_object.log_parameters(mlflow=main_training.mlflow)
            except ValueError:
                continue
            finally:
                # Clean up resources
                main_training.close()
                
            total_amount_model_total += total_amount_model
            total_amount_model_list.append(total_amount_model)

        # Calculate the fitness value
        fitness_value = total_amount_model_total / len(list_seeds)

        # Return the fitness value as a tuple
        return (fitness_value,)

    def main(self, base_config: dict, target_fitness: float = 1000, checkpoint_path: str = None) -> tuple[dict, dict]:
        """
        Run the evolutionary algorithm with checkpointing support.
        
        Args:
            checkpoint_interval: Save a checkpoint every N generations
            checkpoint_path: Path to a checkpoint file to resume from
        """
        
        # Early stopping variables
        min_delta = 50
        no_improvement_generations = 0
        prev_best_fitness = -np.inf
        
        # Check if we're resuming from a checkpoint
        start_gen = 1
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"\nLoading checkpoint from {checkpoint_path}")
            start_gen, population, config_hyperparameters, optimizer_params = load_checkpoint(checkpoint_path)
            self.config_hyperparameters = config_hyperparameters
            self.population_size = optimizer_params['population_size']
            self.generations = optimizer_params['generations']
            self.pc = optimizer_params['pc']
            self.pm = optimizer_params['pm']
            self.mutation_std_dev = optimizer_params['mutation_std_dev']
            self.perform_crossover = optimizer_params['perform_crossover']
            self.patience = optimizer_params['patience']
            self.setup_toolbox()  # Reinitialize toolbox with updated parameters

            # Merge loaded stats with our fresh stats object
            print(f"Resuming from generation {start_gen}")
        else:
            # Initialize fresh population
            print("Generating initial population...")
            population = self.generate_population()
            
            # Evaluate initial population
            for (i, ind) in enumerate(population):
                value_fitness = self.evaluate_individual(ind, copy.deepcopy(base_config), indiv_number=i, gen_number=0)
                if value_fitness == 0:
                    continue
                else:
                    ind.fitness.values = value_fitness
        
        # Main evolutionary loop
        for gen in tqdm(range(start_gen, self.generations + 1)):
            # Initialize tracking variables
            stats = {}
            print(f"\n--- Generation {gen} ---")
            
            # Elitism
            elite_count = min(self.population_size, int((gen / self.generations) * self.population_size)//3)
            print(f"Elite count for generation {gen}: {elite_count}")
            elites = self.toolbox.selection_elite(population, elite_count)
            
            # Selection and variation
            num_parents_needed = self.population_size // 2
            parents = self.toolbox.selection(population, num_parents_needed)
            print(f"selected {len(parents)} parents for generation {gen}")
            
            offspring = []
            num_offspring = int(self.population_size * 1.5)

            if self.perform_crossover:
                while len(offspring) < num_offspring and len(parents) >= 2:
                    # Select two different parents randomly
                    parent1, parent2 = random.sample(parents, 2)
                    
                    # Clone parents before crossover to avoid modifying originals
                    child1, child2 = self.toolbox.clone(parent1), self.toolbox.clone(parent2)

                    # Apply crossover
                    self.toolbox.crossover(child1, child2)
                
                    # Add to offspring
                    offspring.extend([child1, child2])

                # Trim if we got more than needed
                offspring = offspring[:num_offspring]

                # Mutation
                for mutant in offspring:
                    self.toolbox.mutation(mutant)
            else:
                # Mutation until we have enough offspring
                while len(offspring) < num_offspring and len(parents) > 0:
                    # generate lambda / mu offspring per parent
                    offspring_per_parent = num_offspring // len(parents)
                    for parent in parents:
                        for _ in range(offspring_per_parent):
                            mutant = self.toolbox.clone(parent)
                            self.toolbox.mutation(mutant)
                            offspring.append(mutant)

                    # choose random offsprings if there are too many
                    if len(offspring) > num_offspring:
                        offspring = random.sample(offspring, num_offspring)

            # Evaluate offspring
            for (i, ind) in enumerate(offspring):
                value_fitness = self.evaluate_individual(ind, copy.deepcopy(base_config), indiv_number=i, gen_number=gen)
                if value_fitness == 0:
                    continue
                else:
                    ind.fitness.values = value_fitness

            # Perform offspring selection
            offspring = self.toolbox.selection(offspring, self.population_size)

            # Combine populations
            population = elites + offspring

            # Ensure population size is maintained
            assert len(population) == elite_count + self.population_size, "Population size mismatch"
            
            # Update statistics
            fitnesses = [ind.fitness.values[0] for ind in population]
            stats['generations'] = gen
            stats['best_fitness'] = max(fitnesses)
            stats['avg_fitness'] = sum(fitnesses)/len(fitnesses)
            stats['worst_fitness'] = min(fitnesses)
            stats['best_individuals'] = tools.selBest(population, 1)[0]
            stats['population'] = population
            
            # Save checkpoint
            self.save_checkpoint(gen, population, stats)
            
            # Early stopping check
            best_individual = tools.selBest(population, 1)[0]
            best_fitness = best_individual.fitness.values[0]
            
            if best_fitness - prev_best_fitness < min_delta:
                no_improvement_generations += 1
            else:
                no_improvement_generations = 0
                prev_best_fitness = best_fitness
            
            if best_fitness >= target_fitness:
                print(f"Target fitness {target_fitness} reached at generation {gen}. Stopping early.")
                break
            if no_improvement_generations >= self.patience:
                print(f"No improvement for {self.patience} generations. Stopping early.")
                break
    

    def save_checkpoint(self, generation: int, population: list, stats: dict, checkpoint_dir: str = "checkpoints_2") -> str:
        """Save the current state of the optimization to a checkpoint file."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'generation': generation,
            'population': population,
            'stats': stats,
            'random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'timestamp': datetime.now().isoformat(),
            'config_hyperparameters': self.config_hyperparameters,
            'optimizer_params': {
                'population_size': self.population_size,
                'generations': self.generations,
                'pc': self.pc,
                'pm': self.pm,
                'mutation_std_dev': self.mutation_std_dev,
                'patience': self.patience,
                'perform_crossover': self.perform_crossover
            }
        }
        
        filename = os.path.join(checkpoint_dir, f"checkpoint_gen_{generation}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        return filename
    