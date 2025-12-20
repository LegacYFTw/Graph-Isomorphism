import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
from scipy.interpolate import make_interp_spline
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# --- VISUALIZATION CONFIG ---
sns.set_theme(style="white", context="talk")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

class PhaseDiagramAnalyzer:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.data_dir = os.path.join(results_dir, "data")
        self.output_dir = os.path.join(results_dir, "separation_analysis")
        os.makedirs(self.output_dir, exist_ok=True)
        self.matrix_dir = os.path.join(self.output_dir, "confusion_matrices")
        os.makedirs(self.matrix_dir, exist_ok=True)
        
        self.df = self.load_data()

    def load_data(self):
        print("Loading data...")
        records = []
        energy_files = glob.glob(os.path.join(self.data_dir, "*_energies.npy"))
        
        if not energy_files:
            raise FileNotFoundError("No data found.")

        for e_file in energy_files:
            base_name = os.path.basename(e_file).replace("_energies.npy", "")
            energies = np.load(e_file)
            match = re.search(r'_(\d+)nodes', base_name)
            n_nodes = int(match.group(1)) if match else 0
            
            # Ground Truth Logic
            ground_truth = "Non-Isomorphic" if "vs" in base_name else "Isomorphic"
            is_iso = 1 if ground_truth == "Isomorphic" else 0
            
            for energy in energies:
                records.append({
                    'n_nodes': n_nodes,
                    'energy': energy,
                    'Ground_Truth': ground_truth,
                    'is_iso': is_iso
                })
        
        df = pd.DataFrame(records)
        print(f"Loaded {len(df)} samples.")
        return df

    def calculate_dynamic_thresholds(self):
        """Calculates the optimal energy cutoff for each graph size."""
        stats = []
        sizes = sorted(self.df['n_nodes'].unique())
        
        for n in sizes:
            sub = self.df[self.df['n_nodes'] == n]
            if len(sub['Ground_Truth'].unique()) < 2: continue
            
            # ROC Analysis
            fpr, tpr, thresholds = roc_curve(sub['is_iso'], -sub['energy'])
            # Youden's J to find optimal point
            best_idx = np.argmax(tpr - fpr)
            cutoff = -thresholds[best_idx]
            
            # Calculate accuracy at this specific cutoff
            pred = (sub['energy'] <= cutoff).astype(int)
            acc = accuracy_score(sub['is_iso'], pred)
            
            stats.append({'n': n, 'cutoff': cutoff, 'acc': acc})
            
            # Save Confusion Matrix IMMEDIATELY
            self.save_confusion_matrix(sub['is_iso'], pred, n, acc)
            
        return pd.DataFrame(stats)

    def save_confusion_matrix(self, y_true, y_pred, n, acc):
        """Generates and saves a confusion matrix for a specific size."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Iso", "Iso"])
        # Custom color map for clarity
        disp.plot(cmap='Blues', values_format='d', colorbar=False)
        
        plt.title(f"Size N={n} | Accuracy: {acc:.0%}", fontsize=14, fontweight='bold')
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.matrix_dir, f"cm_size_{n}.png"))
        plt.close()

    def plot_phase_separation(self, thresholds_df):
        """
        Visualizes the 'Hyperplane' (Decision Boundary) separating the phases.
        X-axis: Graph Size
        Y-axis: Energy
        """
        print("Generating Phase Separation Plot...")
        plt.figure(figsize=(12, 8))
        
        # 1. Plot the Scatter Data (Background)
        sns.scatterplot(data=self.df, x="n_nodes", y="energy", hue="Ground_Truth",
                        palette={"Isomorphic": "forestgreen", "Non-Isomorphic": "firebrick"},
                        alpha=0.6, s=100, edgecolor="k")
        
        # 2. Plot the Interpolated Decision Boundary (The "Hyperplane")
        if len(thresholds_df) > 1:
            x = thresholds_df['n']
            y = thresholds_df['cutoff']
            
            # Smooth curve generation
            X_Y_Spline = make_interp_spline(x, y)
            X_new = np.linspace(x.min(), x.max(), 500)
            Y_new = X_Y_Spline(X_new)
            
            # Plot the line
            plt.plot(X_new, Y_new, color='navy', linewidth=3, linestyle='--', label='Decision Boundary')
            
            # 3. Shade the Regions (Hyperplane Separation)
            # Shade Green (Iso Zone) below the line
            plt.fill_between(X_new, 0, Y_new, color='green', alpha=0.1, label='Iso Prediction Zone')
            # Shade Red (Non-Iso Zone) above the line
            plt.fill_between(X_new, Y_new, self.df['energy'].max()*1.1, color='red', alpha=0.1, label='Non-Iso Prediction Zone')

        plt.title("Quantum Phase Separation: Isomorphism Decision Boundary", fontsize=16, fontweight='bold')
        plt.xlabel("Graph Size (N Nodes)", fontsize=14)
        plt.ylabel("Final Hamiltonian Energy", fontsize=14)
        plt.legend(loc='upper left', frameon=True, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, self.df['energy'].max() * 1.1)
        
        # Force integer ticks on X axis
        plt.xticks(sorted(self.df['n_nodes'].unique()))
        
        plt.savefig(os.path.join(self.output_dir, "viz_hyperplane_separation.png"))
        plt.close()

    def run(self):
        thresholds = self.calculate_dynamic_thresholds()
        self.plot_phase_separation(thresholds)
        print(f"\nAnalysis Complete.")
        print(f"Confusion Matrices: {self.matrix_dir}")
        print(f"Separation Plot: {self.output_dir}/viz_hyperplane_separation.png")

if __name__ == "__main__":
    base = "graph_isomorphism_results"
    if os.path.exists(base):
        runs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        runs.sort(key=os.path.getmtime)
        if runs:
            latest = runs[-1]
            print(f"Targeting: {latest}")
            PhaseDiagramAnalyzer(latest).run()
        else:
            print("No run data found.")
    else:
        print("Results directory not found.")