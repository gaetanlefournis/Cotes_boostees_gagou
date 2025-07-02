import copy
import os
import pickle
import random
from datetime import datetime

import numpy as np
from deap import base, creator, tools
from tqdm import tqdm

from ai_model.main_ai_model import MainTrainingAIModel
from utils.constants import LIST_SEEDS


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
        **kwargs,
    ) -> None:
        self.config_hyperparameters = config_hyperparameters
        self.population_size = population_size
        self.generations = generations
        self.pc = pc
        self.pm = pm
        self.mutation_std_dev = mutation_std_dev
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
        self.toolbox.register("merge_populations", self.merge_populations)
        self.toolbox.register("clone", copy.deepcopy)

    def generate_individual(self):
        """Generate a random individual based on the hyperparameter configuration."""
        individual = creator.Individual()
        
        # Training parameters
        if 'TRAINING_PARAMETERS' in self.config_hyperparameters.keys():
            training_params = self.config_hyperparameters['TRAINING_PARAMETERS']
            for param, values in training_params.items():
                if values['type'] == 'choice':
                    individual[param] = {'value': random.choice(values['values']), 'type': 'choice', 'values': values['values']}
                elif values['type'] == 'uniform':
                    individual[param] = {'value': random.uniform(values['low'], values['high']), 'type': 'uniform', 'low': values['low'], 'high': values['high']}
                elif values['type'] == 'loguniform':
                    individual[param] = {'value': 10**random.uniform(np.log10(values['low']), np.log10(values['high'])), 'type': 'loguniform', 'low': values['low'], 'high': values['high']}

        # Model parameters
        if 'MODEL_PARAMETERS' in self.config_hyperparameters.keys():
            model_params = self.config_hyperparameters['MODEL_PARAMETERS']
            for param, values in model_params.items():
                if values['type'] == 'choice':
                    individual[param] = {'value': random.choice(values['values']), 'type': 'choice', 'values': values['values']}
                elif values['type'] == 'uniform':
                    individual[param] = {'value': random.uniform(values['low'], values['high']), 'type': 'uniform', 'low': values['low'], 'high': values['high']}
                elif values['type'] == 'loguniform':
                    individual[param] = {'value': 10**random.uniform(np.log10(values['low']), np.log10(values['high'])), 'type': 'loguniform', 'low': values['low'], 'high': values['high']}

        # Data parameters
        if 'DATA_PARAMETERS' in self.config_hyperparameters.keys():
            data_params = self.config_hyperparameters['DATA_PARAMETERS']
            for param, values in data_params.items():
                if values['type'] == 'choice':
                    individual[param] = {'value': random.choice(values['values']), 'type': 'choice', 'values': values['values']}
                elif values['type'] == 'uniform':
                    individual[param] = {'value': random.uniform(values['low'], values['high']), 'type': 'uniform', 'low': values['low'], 'high': values['high']}
                elif values['type'] == 'loguniform':
                    individual[param] = {'value': 10**random.uniform(np.log10(values['low']), np.log10(values['high'])), 'type': 'loguniform', 'low': values['low'], 'high': values['high']}

        # Ensure the output is DEAP-compatible
        if not isinstance(individual, dict):
            raise TypeError("Individual must be a dict")
        print("Generated individual keys:", individual.keys())  # Debug output
        print(f"Generated individual: {individual}")
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
        print(f"intial individual before mutation: {individual}")
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

        print(f"individual after mutation: {individual}")
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

            # Create an instance of MainTrainingAIModel with the updated config
            main_training = MainTrainingAIModel(config=config)

            # Run the training and get the total amount won
            try:
                _, total_amount_model, _ = main_training.run()
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
    
    def merge_populations(self, population1: list[dict], population2: list[dict]) -> list[dict]:
        """Merge two populations into one."""
        merged_population = population1 + population2
        # Ensure the merged population does not exceed the maximum size
        if len(merged_population) > self.population_size:
            merged_population = tools.selBest(merged_population, k=self.population_size)
        return merged_population

    def main(self, base_config: dict, target_fitness: float = 1000, 
            checkpoint_interval: int = 1, checkpoint_path: str = None) -> tuple[dict, dict]:
        """
        Run the evolutionary algorithm with checkpointing support.
        
        Args:
            checkpoint_interval: Save a checkpoint every N generations
            checkpoint_path: Path to a checkpoint file to resume from
        """
        # Initialize tracking variables
        stats = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': [],
            'best_individuals': []
        }
        
        # Early stopping variables
        patience = 8
        min_delta = 50
        no_improvement_generations = 0
        prev_best_fitness = -np.inf
        
        # Check if we're resuming from a checkpoint
        start_gen = 1
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"\nLoading checkpoint from {checkpoint_path}")
            start_gen, population, loaded_stats = self.load_checkpoint(checkpoint_path)
            
            # Merge loaded stats with our fresh stats object
            stats.update(loaded_stats)
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
            print(f"\n--- Generation {gen} ---")
            
            # Elitism
            elite_count = min(self.population_size, int((gen / self.generations) * self.population_size))
            elites = self.toolbox.selection_elite(population, elite_count)
            
            # Selection and variation
            offspring = self.toolbox.selection(population, len(population) - elite_count)
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Crossover
            for i in range(1, len(offspring), 2):
                if i + 1 < len(offspring):
                    self.toolbox.crossover(offspring[i - 1], offspring[i])
            
            # Mutation
            for mutant in offspring:
                self.toolbox.mutation(mutant)
            
            # Evaluate offspring
            for (i, ind) in enumerate(offspring):
                value_fitness = self.evaluate_individual(ind, copy.deepcopy(base_config), indiv_number=i, gen_number=gen)
                if value_fitness == 0:
                    continue
                else:
                    ind.fitness.values = value_fitness
                    
            # Combine populations
            population = self.toolbox.merge_populations(elites, offspring)
            
            # Update statistics
            fitnesses = [ind.fitness.values[0] for ind in population]
            stats['generations'].append(gen)
            stats['best_fitness'].append(max(fitnesses))
            stats['avg_fitness'].append(sum(fitnesses)/len(fitnesses))
            stats['worst_fitness'].append(min(fitnesses))
            stats['best_individuals'].append(tools.selBest(population, 1)[0])
            
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
            
            if best_fitness >= target_fitness or no_improvement_generations >= patience:
                print("Stopping early")
                break
        
        best_individual = tools.selBest(population, 1)[0]
        print("\nBest Individual:")
        for k, v in best_individual.items():
            print(f"{k}: {v['value']}")
        print(f"Fitness: {best_individual.fitness.values[0]:.4f}")
        
        return best_individual, stats
    

    def save_checkpoint(self, generation: int, population: list, stats: dict, checkpoint_dir: str = "checkpoints") -> str:
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
                'mutation_std_dev': self.mutation_std_dev
            }
        }
        
        filename = os.path.join(checkpoint_dir, f"checkpoint_gen_{generation}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Keep only the most recent checkpoint to save space
        for old_file in os.listdir(checkpoint_dir):
            if old_file != os.path.basename(filename):
                os.remove(os.path.join(checkpoint_dir, old_file))
        
        return filename

    def load_checkpoint(self, checkpoint_path: str) -> tuple:
        """Load a saved checkpoint and return the optimization state."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Restore random states
        random.setstate(checkpoint['random_state'])
        np.random.set_state(checkpoint['numpy_random_state'])
        
        return (
            checkpoint['generation'],
            checkpoint['population'],
            checkpoint['stats']
        )
