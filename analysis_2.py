import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
import pickle
from scipy.stats import wasserstein_distance, kstest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import umap
from matplotlib.colors import LinearSegmentedColormap

# --- CONFIGURATION ---
sns.set_theme(style="white", context="paper", font_scale=1.4)
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.5

class SophisticatedAnalyzer:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.data_dir = os.path.join(results_dir, "data")
        self.output_dir = os.path.join(results_dir, "sophisticated_analysis")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.df, self.histories = self.load_data()

    def load_data(self):
        print("Loading data and optimization histories...")
        records = []
        all_histories = []
        energy_files = glob.glob(os.path.join(self.data_dir, "*_energies.npy"))
        
        if not energy_files: raise FileNotFoundError("No data found.")

        for e_file in energy_files:
            base_name = os.path.basename(e_file).replace("_energies.npy", "")
            energies = np.load(e_file)
            
            # Load Histories for Trajectory Analysis
            hist_path = os.path.join(self.data_dir, f"{base_name}_histories.pkl")
            if os.path.exists(hist_path):
                with open(hist_path, 'rb') as f:
                    run_histories = pickle.load(f)
            else:
                run_histories = [[] for _ in energies]

            match = re.search(r'_(\d+)nodes', base_name)
            n_nodes = int(match.group(1)) if match else 0
            ground_truth = "Non-Isomorphic" if "vs" in base_name else "Isomorphic"
            
            for i, energy in enumerate(energies):
                # Interpolate history to fixed length for vectorization
                h = run_histories[i] if i < len(run_histories) and len(run_histories[i]) > 0 else [energy]
                
                records.append({
                    'n_nodes': n_nodes,
                    'energy': energy,
                    'Ground_Truth': ground_truth,
                    'is_iso': 1 if ground_truth == "Isomorphic" else 0,
                    'history_len': len(h)
                })
                all_histories.append(h)
        
        return pd.DataFrame(records), all_histories

    # ==========================================
    # 1. QUANTUM PROBABILITY CONTOURS (GMM)
    # ==========================================
    def analyze_probabilistic_separation(self):
        """
        Uses Gaussian Mixture Models to create a continuous probability map.
        Instead of a hard line, we see where the 'Isomorphic Probability' fades.
        """
        print("Generating GMM Probability Contours...")
        
        # Create a grid for visualization
        n_range = np.linspace(self.df['n_nodes'].min() - 0.5, self.df['n_nodes'].max() + 0.5, 200)
        e_range = np.linspace(0, self.df['energy'].max() * 1.1, 200)
        N_grid, E_grid = np.meshgrid(n_range, e_range)
        Z = np.zeros_like(N_grid)

        # Fit GMM for each integer size N
        for n_val in sorted(self.df['n_nodes'].unique()):
            sub = self.df[self.df['n_nodes'] == n_val]
            if len(sub['Ground_Truth'].unique()) < 2: continue
            
            # Fit GMM: 2 components (Iso vs Non-Iso)
            # We reshape to shape (Samples, 1)
            X = sub['energy'].values.reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(X)
            
            # Identify which component is "Isomorphic" (lower mean energy)
            means = gmm.means_.flatten()
            iso_idx = np.argmin(means)
            
            # Evaluate probability on the Energy grid for this specific N
            # We construct a slice of the grid corresponding to N=n_val
            # (Gaussian weighting for smooth interpolation between sizes)
            prob_iso = gmm.predict_proba(e_range.reshape(-1, 1))[:, iso_idx]
            
            # Spread this probability across the N-axis using a kernel (smooth bands)
            for j, n_grid_val in enumerate(n_range):
                dist = abs(n_grid_val - n_val)
                if dist < 0.8: # Influence radius of 0.8
                    weight = np.exp(-dist**2 / 0.1)
                    Z[:, j] += prob_iso * weight

        # Normalize Z (columns should sum roughly to max weight)
        Z = np.clip(Z, 0, 1)

        # PLOT
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # Custom Diverging Colormap (Green -> White -> Red)
        cmap = LinearSegmentedColormap.from_list("IsoProb", ["#d73027", "#f46d43", "#ffffff", "#66bd63", "#1a9850"])
        
        contour = ax.contourf(N_grid, E_grid, Z, levels=100, cmap=cmap, alpha=0.9)
        cbar = plt.colorbar(contour, label="Probability of Isomorphism $P(Iso|E, N)$")
        
        # Overlay Scatter
        sns.scatterplot(data=self.df, x="n_nodes", y="energy", hue="Ground_Truth", 
                        style="Ground_Truth", palette={"Isomorphic": "green", "Non-Isomorphic": "red"},
                        edgecolor='k', s=80, ax=ax, alpha=0.6)
        
        ax.set_title("Quantum Phase Diagram: Probabilistic Decision Landscape", fontweight='bold')
        ax.set_xlabel("Graph Size ($N$)")
        ax.set_ylabel("Final Energy")
        ax.set_ylim(0, self.df['energy'].max())
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "sophisticated_GMM_contours.png"))
        plt.close()

    # ==========================================
    # 2. STATISTICAL DISTINGUISHABILITY
    # ==========================================
    def analyze_statistical_distance(self):
        """
        Calculates Wasserstein Distance and KS-Test.
        This answers: "Are these distributions mathematically distinct?"
        """
        print("Calculating Statistical Distances...")
        stats = []
        sizes = sorted(self.df['n_nodes'].unique())
        
        for n in sizes:
            sub = self.df[self.df['n_nodes'] == n]
            iso = sub[sub['is_iso'] == 1]['energy']
            non_iso = sub[sub['is_iso'] == 0]['energy']
            
            if len(iso) < 2 or len(non_iso) < 2: continue
            
            # Wasserstein (Earth Mover's) Distance
            w_dist = wasserstein_distance(iso, non_iso)
            
            # Kolmogorov-Smirnov Test (p-value < 0.05 implies distributions are different)
            ks_stat, p_val = kstest(iso, non_iso)
            
            stats.append({
                'Size': n,
                'Wasserstein Distance': w_dist,
                'KS Statistic': ks_stat,
                'P-Value': p_val,
                'Separable': p_val < 0.05
            })
            
        stats_df = pd.DataFrame(stats)
        
        # Visualization of Separability Decay
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Graph Size (N)')
        ax1.set_ylabel('Wasserstein Distance (Separability)', color=color, fontweight='bold')
        ax1.plot(stats_df['Size'], stats_df['Wasserstein Distance'], color=color, marker='o', lw=3)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        # Threshold line for KS test
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('KS Test P-Value (Log Scale)', color=color, fontweight='bold')
        ax2.semilogy(stats_df['Size'], stats_df['P-Value'], color=color, marker='x', linestyle='--', lw=2)
        ax2.axhline(0.05, color='gray', linestyle=':', label='Significance Threshold (0.05)')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title("Statistical Decay of Quantum Advantage")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "sophisticated_statistical_decay.png"))
        plt.close()
        
        stats_df.to_csv(os.path.join(self.output_dir, "statistical_metrics.csv"), index=False)

    # ==========================================
    # 3. TRAJECTORY MANIFOLD (UMAP)
    # ==========================================
    def analyze_trajectories(self):
        """
        Uses UMAP to visualize the entire optimization history.
        Does the 'path' to the solution look different for Isomorphic graphs?
        """
        print("Analyzing Optimization Trajectories (UMAP)...")
        
        # 1. Preprocess Histories: Interpolate to fixed length
        target_len = 50
        processed_histories = []
        valid_indices = []
        
        for idx, h in enumerate(self.histories):
            if len(h) < 2: continue # Skip failed runs
            # Normalize length to target_len
            x_old = np.linspace(0, 1, len(h))
            x_new = np.linspace(0, 1, target_len)
            h_interp = np.interp(x_new, x_old, h)
            
            # Normalize Energy values (0 to 1) to focus on shape
            h_norm = (h_interp - np.min(h_interp)) / (np.max(h_interp) - np.min(h_interp) + 1e-9)
            processed_histories.append(h_norm)
            valid_indices.append(idx)
            
        if not processed_histories:
            print("Not enough history data for UMAP.")
            return

        X_traj = np.array(processed_histories)
        y_traj = self.df.iloc[valid_indices]['Ground_Truth'].values
        n_traj = self.df.iloc[valid_indices]['n_nodes'].values
        
        # 2. Run UMAP
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(X_traj)
        
        # 3. Plot
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=y_traj, style=n_traj,
                        palette={"Isomorphic": "green", "Non-Isomorphic": "red"},
                        s=100, alpha=0.8, edgecolor='k')
        
        plt.title("Manifold Projection of Optimization Trajectories (UMAP)")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "sophisticated_trajectory_manifold.png"))
        plt.close()

    def run(self):
        self.analyze_probabilistic_separation()
        self.analyze_statistical_distance()
        try:
            self.analyze_trajectories()
        except ImportError:
            print("UMAP not installed. Skipping trajectory analysis. (pip install umap-learn)")
        except Exception as e:
            print(f"Trajectory analysis skipped: {e}")
            
        print("\nSophisticated Analysis Complete.")

if __name__ == "__main__":
    base = "graph_isomorphism_results"
    # Auto-find latest run
    if os.path.exists(base):
        runs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        runs.sort(key=os.path.getmtime)
        if runs:
            latest = runs[-1]
            print(f"Targeting: {latest}")
            SophisticatedAnalyzer(latest).run()
        else:
            print("No run data found.")