import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import glob
import json
import re
from sklearn.metrics import roc_curve, auc
from scipy.stats import ttest_ind

# Set professional scientific style
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['savefig.dpi'] = 300

class IsoAnalyzer:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.data_dir = os.path.join(results_dir, "data")
        self.output_dir = os.path.join(results_dir, "analysis_suite")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.df = self.load_data()
        
    def load_data(self):
        """Loads data either from aggregate pickle or by crawling individual files."""
        pkl_path = os.path.join(self.data_dir, "all_results.pkl")
        
        # 1. Try loading the aggregate file first
        if os.path.exists(pkl_path):
            print(f"Loading aggregate data from {pkl_path}...")
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            df = pd.DataFrame(data['results_db'])
            if 'Ground_Truth' not in df.columns:
                df['Ground_Truth'] = df['pair_name'].apply(self._determine_ground_truth)
            return df
            
        # 2. Fallback: Reconstruct from individual files
        print("Aggregate file not found. Reconstructing from individual files...")
        records = []
        
        # Find all energy files
        energy_files = glob.glob(os.path.join(self.data_dir, "*_energies.npy"))
        
        for e_file in energy_files:
            base_name = os.path.basename(e_file).replace("_energies.npy", "")
            energies = np.load(e_file)
            
            # Try to find corresponding JSON for metadata
            json_path = os.path.join(self.data_dir, f"{base_name}_best_result.json")
            metadata = {}
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
            
            # Extract N nodes from filename using regex
            match = re.search(r'_(\d+)nodes', base_name)
            n_nodes = int(match.group(1)) if match else 0
            
            # Determine Truth
            ground_truth = self._determine_ground_truth(base_name)
            
            for i, energy in enumerate(energies):
                records.append({
                    'pair_name': base_name,
                    'n_nodes': n_nodes,
                    'energy': energy,
                    'run_id': i,
                    'Ground_Truth': ground_truth,
                    'shots': metadata.get('metadata', {}).get('shots', 1024)
                })
                
        df = pd.DataFrame(records)
        print(f"Reconstructed {len(df)} runs from {len(energy_files)} graph pairs.")
        
        # Verify we don't have unexpected "Unknowns"
        print("Class Distribution:", df['Ground_Truth'].value_counts().to_dict())
        return df

    def _determine_ground_truth(self, name):
        """
        Correctly identifies ground truth based on generation naming convention.
        - 'vs' indicates comparison of different graphs (Non-Isomorphic)
        - Everything else (Permuted, Wheel, Bipartite, Single) implies Isomorphism
        """
        if "vs" in name:
            return "Non-Isomorphic"
        return "Isomorphic"

    def plot_energy_distributions(self):
        """Global Histogram of Energies."""
        plt.figure(figsize=(10, 6))
        
        # Define palette with fallback for safety
        palette = {
            "Isomorphic": "forestgreen", 
            "Non-Isomorphic": "firebrick",
            "Unknown": "gray"  # Safety color
        }
        
        sns.histplot(data=self.df, x="energy", hue="Ground_Truth", 
                     kde=True, element="step", bins=40, 
                     palette=palette)
        
        plt.title("Global Energy Distribution: Iso vs Non-Iso")
        plt.xlabel("Final Hamiltonian Energy")
        plt.savefig(os.path.join(self.output_dir, "01_global_energy_dist.png"))
        plt.close()

    def plot_box_by_size(self):
        """Box plots separated by Graph Size."""
        plt.figure(figsize=(12, 6))
        palette = {"Isomorphic": "forestgreen", "Non-Isomorphic": "firebrick", "Unknown": "gray"}
        
        sns.boxplot(data=self.df, x="n_nodes", y="energy", hue="Ground_Truth",
                    palette=palette, gap=0.1)
        
        plt.title("Energy Separation by Graph Size")
        plt.ylabel("Final Energy")
        plt.xlabel("Number of Nodes")
        plt.savefig(os.path.join(self.output_dir, "02_energy_by_size_box.png"))
        plt.close()

    def plot_swarm_by_size(self):
        """Swarm plots to see individual data points and overlaps."""
        plt.figure(figsize=(12, 8))
        palette = {"Isomorphic": "forestgreen", "Non-Isomorphic": "firebrick", "Unknown": "gray"}
        
        sns.swarmplot(data=self.df, x="n_nodes", y="energy", hue="Ground_Truth", 
                      dodge=True, size=4, alpha=0.8,
                      palette=palette)
        
        plt.title("Detailed Energy Landscape (Swarm Plot)")
        plt.savefig(os.path.join(self.output_dir, "03_energy_by_size_swarm.png"))
        plt.close()

    def plot_roc_analysis(self):
        """Generates ROC curves and calculates optimal thresholds per size."""
        unique_sizes = sorted(self.df['n_nodes'].unique())
        
        plt.figure(figsize=(10, 8))
        optimal_thresholds = {}
        
        # Use a colormap that handles varying number of sizes gracefully
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_sizes)))
        
        for i, n in enumerate(unique_sizes):
            sub = self.df[self.df['n_nodes'] == n]
            
            # Skip if we don't have both classes for this size
            if len(sub['Ground_Truth'].unique()) < 2:
                print(f"Skipping ROC for N={n}: Not enough class diversity.")
                continue
                
            y_true = (sub['Ground_Truth'] == "Isomorphic").astype(int)
            y_score = -sub['energy'] 
            
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, color=colors[i], 
                     label=f'N={n} (AUC = {roc_auc:.2f})')
            
            # Youden's J
            J = tpr - fpr
            ix = np.argmax(J)
            best_thresh = -thresholds[ix]
            optimal_thresholds[n] = best_thresh

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves by Graph Size')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.output_dir, "04_roc_curves.png"))
        plt.close()
        
        return optimal_thresholds

    def generate_classification_map(self, thresholds):
        """Generates a decision boundary plot."""
        if not thresholds:
            print("Skipping decision boundary map (no thresholds calculated).")
            return

        plt.figure(figsize=(10, 6))
        palette = {"Isomorphic": "green", "Non-Isomorphic": "red", "Unknown": "gray"}
        
        sns.scatterplot(data=self.df, x="n_nodes", y="energy", hue="Ground_Truth",
                        palette=palette, alpha=0.3)
        
        sizes = sorted(thresholds.keys())
        thresh_values = [thresholds[s] for s in sizes]
        plt.plot(sizes, thresh_values, 'k--', linewidth=2, label='Optimal Decision Boundary')
        
        plt.title("Classification Decision Boundary")
        plt.xlabel("Graph Size (N)")
        plt.ylabel("Energy")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "05_decision_boundary.png"))
        plt.close()

    def generate_report(self, thresholds):
        """Writes a detailed text report."""
        report_path = os.path.join(self.output_dir, "statistical_report.txt")
        with open(report_path, "w") as f:
            f.write("QUANTUM GRAPH ISOMORPHISM: STATISTICAL REPORT\n")
            f.write("=============================================\n\n")
            
            for n in sorted(self.df['n_nodes'].unique()):
                sub = self.df[self.df['n_nodes'] == n]
                iso = sub[sub['Ground_Truth'] == 'Isomorphic']['energy']
                non_iso = sub[sub['Ground_Truth'] == 'Non-Isomorphic']['energy']
                
                f.write(f"--- Graph Size N={n} ---\n")
                
                if n in thresholds:
                    f.write(f"  Optimal Threshold: < {thresholds[n]:.6f}\n")
                    # Accuracy
                    pred_iso = sub['energy'] < thresholds[n]
                    actual_iso = sub['Ground_Truth'] == 'Isomorphic'
                    acc = (pred_iso == actual_iso).mean()
                    f.write(f"  Classification Accuracy: {acc:.2%}\n")
                
                f.write(f"  Iso Mean (std):     {iso.mean():.6f} ({iso.std():.6f})\n")
                f.write(f"  Non-Iso Mean (std): {non_iso.mean():.6f} ({non_iso.std():.6f})\n")
                
                if len(iso) > 1 and len(non_iso) > 1:
                    # Welch's t-test (does not assume equal variance)
                    t_stat, p_val = ttest_ind(iso, non_iso, equal_var=False)
                    f.write(f"  P-Value (Distinction):   {p_val:.4e}\n")
                f.write("\n")
                
        print(f"Report saved to {report_path}")

    def run(self):
        print(f"Analyzing data in {self.results_dir}...")
        self.plot_energy_distributions()
        self.plot_box_by_size()
        self.plot_swarm_by_size()
        
        thresholds = self.plot_roc_analysis()
        self.generate_classification_map(thresholds)
        self.generate_report(thresholds)
        print("Analysis pipeline complete.")

if __name__ == "__main__":
    base = "graph_isomorphism_results"
    
    if os.path.exists(base):
        runs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        runs.sort(key=os.path.getmtime)
        
        if runs:
            latest_run = runs[-1]
            print(f"Targeting latest run: {latest_run}")
            analyzer = IsoAnalyzer(latest_run)
            analyzer.run()
        else:
            print("No run directories found.")
    else:
        print(f"Base directory '{base}' not found.")