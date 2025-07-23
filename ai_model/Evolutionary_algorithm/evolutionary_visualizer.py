import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm


class EvolutionaryVisualizer:
    def __init__(self, stats: Dict, config_hyperparameters: Dict, output_dir: str = "evolution_plots"):
        """
        Initialize the visualizer with optimization statistics and hyperparameter configuration.
        
        Args:
            stats: Dictionary containing optimization statistics
            config_hyperparameters: Original hyperparameter configuration
            output_dir: Directory to save plots
        """
        self.stats = stats
        self.config_hyperparameters = config_hyperparameters
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style with better default sizes
        plt.style.use('seaborn-v0_8')
        self.colors = cm.viridis(np.linspace(0, 1, 8))
        sns.set_palette(self.colors)
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        
        # Store parameter metadata for smart scaling
        self.param_metadata = self._extract_parameter_metadata()
        
    def _extract_parameter_metadata(self) -> Dict:
        """Extract parameter types and ranges for proper scaling."""
        metadata = {}
        
        for section in ['TRAINING_PARAMETERS', 'MODEL_PARAMETERS', 'DATA_PARAMETERS']:
            if section in self.config_hyperparameters:
                for param, config in self.config_hyperparameters[section].items():
                    metadata[param] = {
                        'type': config['type'],
                        'range': (config.get('low', 0), config.get('high', 1))
                    }
        return metadata
    
    def _smart_scale(self, values: np.ndarray, param: str) -> np.ndarray:
        """
        Apply appropriate scaling based on parameter type and range.
        
        Args:
            values: Array of parameter values
            param: Parameter name
            
        Returns:
            Scaled values suitable for visualization
        """
        if param not in self.param_metadata:
            return values
            
        meta = self.param_metadata[param]
        
        if meta['type'] == 'loguniform':
            # Log transform then scale to [0,1]
            log_values = np.log10(values)
            log_min = np.log10(meta['range'][0])
            log_max = np.log10(meta['range'][1])
            return (log_values - log_min) / (log_max - log_min)
        elif meta['type'] == 'uniform':
            # Linear scale to [0,1]
            min_val, max_val = meta['range']
            return (values - min_val) / (max_val - min_val)
        else:  # choice or unknown
            return values
    
    def plot_fitness_progression(self, log_scale: bool = False) -> None:
        """
        Plot fitness progression with optional log scale.
        
        Args:
            log_scale: Whether to use logarithmic y-axis
        """
        plt.figure(figsize=(12, 6))
        
        generations = self.stats['generations']
        print(self.stats['best_fitness'])
        best_fitness = np.array(self.stats['best_fitness'])
        avg_fitness = np.array(self.stats['avg_fitness'])
        worst_fitness = np.array(self.stats['worst_fitness'])
        
        
        plt.plot(generations, best_fitness, 'o-', label='Best Fitness', linewidth=2)
        plt.plot(generations, avg_fitness, 's-', label='Average Fitness', linewidth=2)
        plt.plot(generations, worst_fitness, 'd-', label='Worst Fitness', linewidth=2)
        
        plt.fill_between(generations, best_fitness, avg_fitness, alpha=0.1)
        plt.fill_between(generations, avg_fitness, worst_fitness, alpha=0.1)
        
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Fitness Value', fontsize=12)
        plt.title('Fitness Progression Across Generations', fontsize=14)
        
        if log_scale:
            plt.yscale('log')
            plt.ylabel('Fitness Value (log scale)', fontsize=12)
        
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fitness_progression.png'), dpi=300)
        plt.close()
    
    def plot_parameter_distributions(self, selected_params: Optional[List[str]] = None) -> None:
        """
        Plot parameter distributions with smart scaling for wide ranges.
        """
        if selected_params is None:
            first_individual = self.stats['population'][0][0]
            selected_params = [p for p in first_individual.keys() if p in self.param_metadata]
        
        n_params = len(selected_params)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_params == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, param in enumerate(selected_params):
            ax = axes[i]
            all_values = []
            generations = []
            
            for gen_idx, pop in enumerate(self.stats['population']):
                for ind in pop:
                    if param in ind:
                        all_values.append(ind[param]['value'])
                        generations.append(gen_idx)
            
            df = pd.DataFrame({'Generation': generations, 'Value': all_values})
            
            if self.param_metadata[param]['type'] == 'choice':
                # Categorical parameter
                sns.countplot(data=df, x='Generation', hue='Value', ax=ax)
                ax.set_title(f'{param} (Categorical)')
                ax.legend(title='Value', bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                # Numerical parameter
                if self.param_metadata[param]['type'] == 'loguniform':
                    ax.set_yscale('log')
                
                sns.boxplot(data=df, x='Generation', y='Value', ax=ax)
                ax.set_title(f'{param} ({self.param_metadata[param]["type"]})')
                
                # Add median line
                medians = df.groupby('Generation')['Value'].median()
                ax.plot(range(len(medians)), medians, color='red', linestyle='--', alpha=0.7)
            
            ax.set_xlabel('Generation')
            ax.set_ylabel('Parameter Value')
        
        # Hide unused axes
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parameter_distributions.png'), dpi=300)
        plt.close()
    
    def plot_parallel_coordinates(self, selected_params: Optional[List[str]] = None) -> None:
        """
        Parallel coordinates plot with independent scaling for each parameter.
        Each parameter axis is normalized to [0,1] based on its own min/max across all generations.
        """
        if len(self.stats['generations']) < 2:
            print("Need at least 2 generations for parallel coordinates plot")
            return
            
        if selected_params is None:
            first_individual = self.stats['population'][0][0]
            selected_params = [p for p in first_individual.keys() if p in self.param_metadata]
        
        # First collect all values to determine scaling ranges
        param_values = {p: [] for p in selected_params}
        for pop in self.stats['population']:
            for ind in pop:
                for param in selected_params:
                    if param in ind:
                        param_values[param].append(ind[param]['value'])
        
        # Calculate min/max for each parameter
        param_ranges = {}
        for param, values in param_values.items():
            if values:  # Only if we have values
                min_val = min(values)
                max_val = max(values)
                # Handle case where min == max (avoid division by zero)
                if min_val == max_val:
                    param_ranges[param] = (min_val, max_val + 1e-10)  # Small offset
                else:
                    param_ranges[param] = (min_val, max_val)
        
        # Prepare data with normalization
        data = []
        for gen_idx, pop in enumerate(self.stats['population']):
            for ind in pop:
                entry = {'Generation': gen_idx, 'Fitness': ind.fitness.values[0]}
                for param in selected_params:
                    if param in ind:
                        val = ind[param]['value']
                        pmin, pmax = param_ranges[param]
                        # Normalize to [0,1] range
                        norm_val = (val - pmin) / (pmax - pmin) if (pmax - pmin) != 0 else 0.5
                        entry[param] = norm_val
                data.append(entry)

        # also normalize fitness column
        fitness_values = [entry['Fitness'] for entry in data]
        fitness_min = min(fitness_values)
        fitness_max = max(fitness_values)
        if fitness_min == fitness_max:
            fitness_min, fitness_max = fitness_min - 1e-10, fitness_max + 1e-10
        for entry in data:
            entry['Fitness'] = (entry['Fitness'] - fitness_min) / (fitness_max - fitness_min)

        # Create DataFrame for plotting
        df = pd.DataFrame(data)
        
        # Create plot with proper scaling
        plt.figure(figsize=(15, 8))
        
        # Get unique generations for coloring
        unique_gens = df['Generation'].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_gens)))
        
        # Plot each generation separately
        for gen, color in zip(unique_gens, colors):
            subset = df[df['Generation'] == gen]
            if len(subset) > 0:
                pd.plotting.parallel_coordinates(
                    subset, 
                    'Generation', 
                    cols=selected_params + ['Fitness'],
                    color=[color] * len(subset),
                    alpha=0.1,
                    linewidth=1
                )
        
        plt.title('Parallel Coordinates Plot (Normalized Parameters)', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.1)
        
        # Adjust y-ticks to show original values on secondary axis
        ax = plt.gca()
        for i, param in enumerate(selected_params + ['Fitness']):
            if param in param_ranges:
                pmin, pmax = param_ranges[param]
                # Create secondary axis with original values
                ax2 = ax.twinx()
                ax2.set_ylim(ax.get_ylim())  # Match normalized scale
                # Position the secondary axis
                ax2.spines['right'].set_position(('axes', i/(len(selected_params))))
                # Set ticks showing original values
                tick_positions = np.linspace(0, 1, 5)
                tick_values = np.linspace(pmin, pmax, 5)
                ax2.set_yticks(tick_positions)
                ax2.set_yticklabels([f"{x:.2e}" if 0 < abs(x) < 1e-2 else f"{x:.2f}" for x in tick_values])
                ax2.set_ylabel(param)
        
        # Create custom legend for generations
        handles = [plt.Line2D([0], [0], color=color, lw=2) for color in colors]
        plt.legend(handles, [f'Gen {g}' for g in unique_gens], title='Generation')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parallel_coordinates.png'), dpi=300)
        plt.close()
    
    def plot_3d_parameter_space(self, param1: str, param2: str, param3: str) -> None:
        """
        3D scatter plot with smart axis scaling.
        """
        # Prepare data
        data = []
        for gen_idx, pop in enumerate(self.stats['population']):
            for ind in pop:
                if all(p in ind for p in [param1, param2, param3]):
                    entry = {
                        'Generation': gen_idx,
                        'Fitness': ind.fitness.values[0],
                        param1: ind[param1]['value'],
                        param2: ind[param2]['value'],
                        param3: ind[param3]['value']
                    }
                    data.append(entry)
                    
        if not data:
            print(f"Missing parameters in individuals")
            return
            
        df = pd.DataFrame(data)
        
        # Create figure with 3D projection
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color by generation
        sc = ax.scatter(
            self._smart_scale(df[param1], param1),
            self._smart_scale(df[param2], param2),
            self._smart_scale(df[param3], param3),
            c=df['Generation'],
            cmap='viridis',
            s=50,
            alpha=0.7
        )
        
        # Set axis labels with scaling info
        ax.set_xlabel(f'{param1}\n({self.param_metadata[param1]["type"]})', fontsize=10)
        ax.set_ylabel(f'{param2}\n({self.param_metadata[param2]["type"]})', fontsize=10)
        ax.set_zlabel(f'{param3}\n({self.param_metadata[param3]["type"]})', fontsize=10)
        plt.title('3D Parameter Space (Normalized Coordinates)', fontsize=14)
        
        # Add colorbar for generation
        cbar = plt.colorbar(sc)
        cbar.set_label('Generation', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'3d_parameter_space_{param1}_{param2}_{param3}.png'), dpi=300)
        plt.close()
    
    def plot_parameter_importance(self, method: str = 'mutual_info') -> None:
        """
        Enhanced parameter importance plot with different calculation methods.
        
        Args:
            method: 'mutual_info', 'pearson', or 'spearman'
        """
        first_individual = self.stats['population'][0][0]
        all_params = [p for p in first_individual.keys() if p in self.param_metadata]
        
        # Prepare data
        data = []
        for pop in self.stats['population']:
            for ind in pop:
                entry = {'Fitness': ind.fitness.values[0]}
                for param in all_params:
                    if param in ind:
                        entry[param] = ind[param]['value']
                data.append(entry)
                
        df = pd.DataFrame(data)
        
        # Calculate importance based on selected method
        if method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            X = df[all_params]
            y = df['Fitness']
            importance = mutual_info_regression(X, y)
            importance = pd.Series(importance, index=all_params).sort_values(ascending=False)
            ylabel = 'Mutual Information with Fitness'
        elif method == 'pearson':
            importance = df.corr()['Fitness'].abs().drop('Fitness').sort_values(ascending=False)
            ylabel = 'Absolute Pearson Correlation'
        elif method == 'spearman':
            importance = df.corr(method='spearman')['Fitness'].abs().drop('Fitness').sort_values(ascending=False)
            ylabel = 'Absolute Spearman Correlation'
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Plot
        plt.figure(figsize=(12, 6))
        importance.plot(kind='bar', color=self.colors)
        plt.title(f'Parameter Importance ({method.capitalize()} Method)', fontsize=14)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'parameter_importance_{method}.png'), dpi=300)
        plt.close()
    
    def generate_all_plots(self) -> None:
        """Generate all plots with appropriate scaling."""
        print("Generating fitness progression plot (log scale)...")
        self.plot_fitness_progression(log_scale=False)
        
        print("Generating parameter distributions plot...")
        self.plot_parameter_distributions()
        
        print("Generating parallel coordinates plot (normalized)...")
        self.plot_parallel_coordinates()
        
        print("Generating parameter importance plots...")
        for method in ['mutual_info', 'pearson', 'spearman']:
            try:
                self.plot_parameter_importance(method=method)
            except Exception as e:
                print(f"Could not generate {method} importance plot: {e}")
        
        # Generate 3D plot if we have at least 3 numerical parameters
        numerical_params = [
            p for p, meta in self.param_metadata.items() 
            if meta['type'] in ['uniform', 'loguniform']
        ]
        if len(numerical_params) >= 3:
            print("Generating 3D parameter space plot...")
            self.plot_3d_parameter_space(numerical_params[0], numerical_params[1], numerical_params[2])
        
        print(f"All plots saved to {self.output_dir}")