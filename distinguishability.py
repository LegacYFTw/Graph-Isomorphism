import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
from scipy.stats import ttest_ind

# --- CONFIGURATION ---
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

class DistinguishabilityVerifier:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.data_dir = os.path.join(results_dir, "data")
        self.output_dir = os.path.join(results_dir, "distinguishability_proof")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.df = self.load_data()

    def load_data(self):
        print("Loading data for verification...")
        records = []
        energy_files = glob.glob(os.path.join(self.data_dir, "*_energies.npy"))
        
        for e_file in energy_files:
            base_name = os.path.basename(e_file).replace("_energies.npy", "")
            energies = np.load(e_file)
            
            match = re.search(r'_(\d+)nodes', base_name)
            n_nodes = int(match.group(1)) if match else 0
            
            # Ground Truth Logic
            ground_truth = "Non-Isomorphic" if "vs" in base_name else "Isomorphic"
            
            for energy in energies:
                records.append({
                    'n_nodes': n_nodes,
                    'energy': energy,
                    'Ground_Truth': ground_truth
                })
        
        return pd.DataFrame(records)

    def verify_n5_separation(self):
        """
        Specifically targets N=5 to prove distinct distributions.
        """
        print("\n--- Verifying Distinguishability for N >= 5 ---")
        
        # Filter for N >= 5
        target_df = self.df[self.df['n_nodes'] >= 5]
        
        if target_df.empty:
            print("No data found for N >= 5.")
            return

        unique_sizes = sorted(target_df['n_nodes'].unique())
        
        for n in unique_sizes:
            sub = target_df[target_df['n_nodes'] == n]
            iso = sub[sub['Ground_Truth'] == 'Isomorphic']['energy']
            non_iso = sub[sub['Ground_Truth'] == 'Non-Isomorphic']['energy']
            
            print(f"\nAnalyzing N={n}:")
            print(f"  Iso Mean: {iso.mean():.6f} (std: {iso.std():.6f})")
            print(f"  Non-Iso Mean: {non_iso.mean():.6f} (std: {non_iso.std():.6f})")
            
            # 1. Calculate Gap
            gap = non_iso.mean() - iso.mean()
            print(f"  Energy Gap: {gap:.6f}")
            
            # 2. Statistical Test (Welch's t-test)
            t_stat, p_val = ttest_ind(iso, non_iso, equal_var=False)
            print(f"  P-Value: {p_val:.4e}")
            
            if p_val < 0.05:
                print("  >> RESULT: Statistically SIGNIFICANT Separation.")
            else:
                print("  >> RESULT: Indistinguishable.")

            # 3. Plot Density (The Visual Proof)
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=sub, x="energy", hue="Ground_Truth", fill=True, 
                        palette={"Isomorphic": "forestgreen", "Non-Isomorphic": "firebrick"},
                        alpha=0.4, linewidth=2)
            
            plt.title(f"Energy Distinguishability at N={n} (Gap = {gap:.4f})", fontweight='bold')
            plt.xlabel("Final Hamiltonian Energy")
            plt.axvline(iso.mean(), color='green', linestyle='--', alpha=0.5)
            plt.axvline(non_iso.mean(), color='red', linestyle='--', alpha=0.5)
            
            # Annotate P-Value
            plt.annotate(f"p < {p_val:.1e}", xy=(0.05, 0.9), xycoords='axes fraction', 
                         fontsize=12, bbox=dict(boxstyle="round", fc="white", ec="black"))

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"proof_separation_N{n}.png"))
            plt.close()

if __name__ == "__main__":
    base = "graph_isomorphism_results"
    if os.path.exists(base):
        runs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        runs.sort(key=os.path.getmtime)
        if runs:
            latest = runs[-1]
            print(f"Targeting: {latest}")
            DistinguishabilityVerifier(latest).verify_n5_separation()
        else:
            print("No run data found.")