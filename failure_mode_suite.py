import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
import pickle

# --- CONFIGURATION ---
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = [12, 7]
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

class EmpiricalFailureAnalyzer:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.data_dir = os.path.join(results_dir, "data")
        self.output_dir = os.path.join(results_dir, "empirical_failure_evidence")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.df = self.load_data()

    def load_data(self):
        print("Mining existing data for failure signals...")
        records = []
        # Find all energy files
        energy_files = glob.glob(os.path.join(self.data_dir, "*_energies.npy"))
        
        if not energy_files:
            raise FileNotFoundError(f"No data found in {self.data_dir}")

        for e_file in energy_files:
            base_name = os.path.basename(e_file).replace("_energies.npy", "")
            energies = np.load(e_file)
            
            # Load History to calculate Optimization Gain
            hist_path = os.path.join(self.data_dir, f"{base_name}_histories.pkl")
            if os.path.exists(hist_path):
                with open(hist_path, 'rb') as f:
                    histories = pickle.load(f)
            else:
                histories = [[] for _ in energies]

            # Parse Metadata
            match = re.search(r'_(\d+)nodes', base_name)
            n_nodes = int(match.group(1)) if match else 0
            ground_truth = "Non-Isomorphic" if "vs" in base_name else "Isomorphic"
            
            for i, energy in enumerate(energies):
                # Calculate Optimization Gain (Start - End)
                # If history exists, use first point. Else assume 0 gain.
                start_energy = histories[i][0] if (i < len(histories) and len(histories[i]) > 0) else energy
                gain = start_energy - energy
                
                records.append({
                    'n_nodes': n_nodes,
                    'energy': energy,
                    'start_energy': start_energy,
                    'optimization_gain': gain,
                    'Ground_Truth': ground_truth
                })
        
        df = pd.DataFrame(records)
        print(f"Loaded {len(df)} samples.")
        return df

    def plot_spectral_collapse(self):
        """
        Evidence of Locality Failure / Spectral Crowding.
        Plots the 'Distinguishability Gap' vs N.
        """
        print("Generating Spectral Collapse Evidence...")
        
        # Calculate Mean Energy per Group
        summary = self.df.groupby(['n_nodes', 'Ground_Truth'])['energy'].mean().unstack()
        
        # Calculate the Gap (Non-Iso - Iso)
        if 'Isomorphic' in summary.columns and 'Non-Isomorphic' in summary.columns:
            summary['Gap'] = summary['Non-Isomorphic'] - summary['Isomorphic']
            
            plt.figure(figsize=(10, 6))
            
            # Plot the Gap
            plt.plot(summary.index, summary['Gap'], 'o--', color='crimson', linewidth=3, markersize=10, label='Energy Gap')
            
            # Add a "Noise Floor" reference line (approximate)
            noise_floor = 0.05 
            plt.axhline(noise_floor, color='gray', linestyle=':', label='Approx. Noise Floor')
            
            # Fill area to show collapse
            plt.fill_between(summary.index, 0, summary['Gap'], color='crimson', alpha=0.1)
            
            plt.title("Empirical Evidence of Spectral Collapse", fontweight='bold')
            plt.xlabel("Graph Size ($N$)")
            plt.ylabel("Energy Gap $\Delta E$ (Signal Strength)")
            plt.legend()
            
            # --- FIX: Use Relative Axes Coordinates for text placement ---
            if len(summary) > 1:
                # Point to the last data point
                target_x = summary.index.max()
                target_y = summary['Gap'].iloc[-1]
                
                plt.annotate("Signal Vanishes\n(Indistinguishable)", 
                             xy=(target_x, target_y),
                             # Place text at 70% x, 80% y of the axis box
                             xytext=(0.7, 0.8), 
                             textcoords='axes fraction',
                             arrowprops=dict(facecolor='black', shrink=0.05, connectionstyle="arc3,rad=-0.2"),
                             horizontalalignment='center')

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "empirical_spectral_collapse.png"))
            plt.close()
        else:
            print("Skipping Spectral Collapse: Data missing both Iso and Non-Iso classes for comparison.")

    def plot_optimization_stagnation(self):
        """
        Evidence of Barren Plateaus / Optimizer Failure.
        Plots Optimization Gain vs N. Decreasing gain = harder landscape.
        """
        print("Generating Optimizer Stagnation Evidence...")
        
        plt.figure(figsize=(10, 6))
        
        sns.lineplot(data=self.df, x="n_nodes", y="optimization_gain", hue="Ground_Truth",
                     palette={"Isomorphic": "green", "Non-Isomorphic": "red"}, 
                     marker="o", linewidth=3)
        
        plt.title("Empirical Evidence of Optimization Stagnation", fontweight='bold')
        plt.xlabel("Graph Size ($N$)")
        plt.ylabel("Optimization Gain (Initial $E$ - Final $E$)")
        
        # --- FIX: Use Relative Axes Coordinates for text placement ---
        # Find point to annotate (max N, approximate mean gain at max N for clarity)
        max_n = self.df['n_nodes'].max()
        # Target the mean of Isomorphic (usually the baseline) at max N
        target_data = self.df[(self.df['n_nodes'] == max_n) & (self.df['Ground_Truth'] == 'Isomorphic')]
        if not target_data.empty:
            target_y = target_data['optimization_gain'].mean()
        else:
            # Fallback to absolute minimum if specific data isn't available
            target_y = self.df['optimization_gain'].min()

        plt.annotate("Diminishing Returns\n(Flat Landscape)", 
                     xy=(max_n, target_y),
                     # Place text at 80% x, 50% y of the axis box
                     xytext=(0.8, 0.5),
                     textcoords='axes fraction',
                     arrowprops=dict(facecolor='black', shrink=0.05, connectionstyle="arc3,rad=0.2"),
                     horizontalalignment='center')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "empirical_optimizer_stagnation.png"))
        plt.close()

    def run(self):
        self.plot_spectral_collapse()
        self.plot_optimization_stagnation()
        print(f"Evidence generated in: {self.output_dir}")

if __name__ == "__main__":
    # Auto-detect latest run
    base = "graph_isomorphism_results"
    if os.path.exists(base):
        runs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        runs.sort(key=os.path.getmtime)
        if runs:
            latest = runs[-1]
            print(f"Targeting: {latest}")
            EmpiricalFailureAnalyzer(latest).run()
        else:
            print("No run data found.")
    else:
        print("Results directory not found.")