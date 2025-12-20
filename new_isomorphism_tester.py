# %%
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import product, combinations
from tqdm import tqdm
import pandas as pd
from scipy import stats
import os
import json
from datetime import datetime
import pickle
from sklearn.decomposition import PCA

# Qiskit Imports
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# %%
# Create output directories
def setup_output_directories(base_dir="graph_isomorphism_results"):
    """Create organized directory structure for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, f"run_{timestamp}")
    
    dirs = {
        'base': results_dir,
        'graph_pairs': os.path.join(results_dir, "graph_pairs"),
        'energy_distributions': os.path.join(results_dir, "energy_distributions"),
        'convergence_plots': os.path.join(results_dir, "convergence_plots"),
        'mapping_matrices': os.path.join(results_dir, "mapping_matrices"),
        'summary_plots': os.path.join(results_dir, "summary_plots"),
        'data': os.path.join(results_dir, "data"),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

# %%
# Define Graph Utilities with saving capability
def save_graph_pair(G1, G2, filename, title1="Graph 1", title2="Graph 2", 
                    main_title="", dpi=300, figsize=(10, 4)):
    """Save two graphs side by side to file."""
    plt.figure(figsize=figsize, dpi=dpi)
    plt.suptitle(main_title, fontsize=14, fontweight='bold')
    
    plt.subplot(1, 2, 1)
    pos1 = nx.spring_layout(G1, seed=42)
    nx.draw(G1, pos1, with_labels=True, node_color='lightblue', 
            edge_color='gray', node_size=500, font_size=10)
    plt.title(title1, fontsize=12)
    
    plt.subplot(1, 2, 2)
    pos2 = nx.spring_layout(G2, seed=42)
    nx.draw(G2, pos2, with_labels=True, node_color='lightgreen', 
            edge_color='gray', node_size=500, font_size=10)
    plt.title(title2, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

def save_adjacency_matrices(G1, G2, filename, dpi=300):
    """Save adjacency matrices side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi)
    
    # Get sorted nodes for consistent ordering
    nodes1 = sorted(G1.nodes())
    nodes2 = sorted(G2.nodes())
    
    # Create adjacency matrices
    A1 = nx.to_numpy_array(G1, nodelist=nodes1)
    A2 = nx.to_numpy_array(G2, nodelist=nodes2)
    
    # Plot matrices
    im1 = ax1.imshow(A1, cmap='Blues', vmin=0, vmax=1)
    ax1.set_title("Graph 1 Adjacency", fontsize=12)
    ax1.set_xticks(range(len(nodes1)))
    ax1.set_yticks(range(len(nodes1)))
    ax1.set_xticklabels(nodes1)
    ax1.set_yticklabels(nodes1)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    im2 = ax2.imshow(A2, cmap='Greens', vmin=0, vmax=1)
    ax2.set_title("Graph 2 Adjacency", fontsize=12)
    ax2.set_xticks(range(len(nodes2)))
    ax2.set_yticks(range(len(nodes2)))
    ax2.set_xticklabels(nodes2)
    ax2.set_yticklabels(nodes2)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

# %%
def get_gi_qubo(G1, G2, penalty_weight=10.0):
    """
    Constructs the Quadratic Program for Graph Isomorphism.
    Variables x_{u,v} = 1 if node u in G1 maps to node v in G2.
    Constraints:
    1. Bijection: One-to-one mapping (Row sums = 1, Col sums = 1).
    2. Edge Consistency: Adjacency matrices must match under permutation.
    """
    n = len(G1.nodes)
    if len(G2.nodes) != n:
        raise ValueError("Graphs must have the same number of nodes")

    qp = QuadraticProgram()
    
    # 1. Create Binary Variables x_{i,j} for mapping node i of G1 to node j of G2
    vars_map = {}
    for u in range(n):
        for v in range(n):
            var_name = f"x_{u}_{v}"
            qp.binary_var(var_name)
            vars_map[(u, v)] = var_name

    # 2. Add Linear constraints for Bijection (One-hot rows and columns)
    # Each node in G1 maps to exactly one node in G2
    for u in range(n):
        qp.linear_constraint(
            linear={vars_map[(u, v)]: 1 for v in range(n)},
            sense="==",
            rhs=1,
            name=f"row_{u}"
        )

    # Each node in G2 is mapped to by exactly one node in G1
    for v in range(n):
        qp.linear_constraint(
            linear={vars_map[(u, v)]: 1 for u in range(n)},
            sense="==",
            rhs=1,
            name=f"col_{v}"
        )

    # 3. Add Quadratic Objective for Edge Inconsistency
    # Minimize: sum (x_{u,v} * x_{u',v'}) for all mappings that violate adjacency
    # Violation: (u, u') is edge in G1 but (v, v') is NOT in G2 (or vice versa)
    
    quad_dict = {}
    
    # Iterate over all pairs of edges potential mappings
    # For every pair of nodes (u, u') in G1 and (v, v') in G2
    for u, up in product(range(n), repeat=2):
        if u == up: continue # Skip self loops for simplicity unless graphs have them
        
        for v, vp in product(range(n), repeat=2):
            if v == vp: continue

            # Check Adjacency
            g1_edge = G1.has_edge(u, up)
            g2_edge = G2.has_edge(v, vp)

            # If structural mismatch found between these pairs
            if g1_edge != g2_edge:
                # We penalize selecting BOTH x_{u,v} and x_{up, vp}
                # Cost += x_{u,v} * x_{up, vp}
                
                key = (vars_map[(u, v)], vars_map[(up, vp)])
                # Sort key to ensure uniqueness in dictionary
                if key[0] > key[1]: key = (key[1], key[0])
                
                quad_dict[key] = quad_dict.get(key, 0) + penalty_weight

    qp.minimize(quadratic=quad_dict)
    return qp

# %%
class GraphIsoAnalyzer:
    def __init__(self, output_dirs):
        self.sampler = StatevectorSampler(seed=42)
        self.results_db = []  # Store all run results
        self.graph_pairs = []  # Store graph pairs
        self.output_dirs = output_dirs
        self.experiment_metadata = {
            'start_time': datetime.now().isoformat(),
            'qiskit_version': '0.45.0',  # Update as needed
            'sampler': 'StatevectorSampler',
            'optimizer': 'COBYLA',
            'algorithm': 'QAOA'
        }
        
    def save_metadata(self):
        """Save experiment metadata."""
        metadata_file = os.path.join(self.output_dirs['data'], 'experiment_metadata.json')
        metadata = {
            **self.experiment_metadata,
            'end_time': datetime.now().isoformat(),
            'total_runs': len(self.results_db),
            'unique_pairs': len(set(r['pair_name'] for r in self.results_db)),
            'graph_sizes': sorted(set(r['n_nodes'] for r in self.results_db))
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_graph_data(self, G1, G2, pair_name):
        """Save graph data in multiple formats."""
        # Create sanitized filename
        safe_name = pair_name.replace(" ", "_").replace(":", "").replace("/", "_")
        base_path = os.path.join(self.output_dirs['graph_pairs'], safe_name)
        
        # Save graph pair visualization
        save_graph_pair(G1, G2, f"{base_path}_graphs.png", 
                       title1="Graph A", title2="Graph B", 
                       main_title=f"{pair_name}\n(n={len(G1.nodes)})")
        
        # Save adjacency matrices
        save_adjacency_matrices(G1, G2, f"{base_path}_adjacency.png")
        
        # Save graph properties
        properties = {
            'pair_name': pair_name,
            'n_nodes': len(G1.nodes),
            'graph1_properties': {
                'nodes': list(G1.nodes()),
                'edges': list(G1.edges()),
                'degree_sequence': sorted([d for n, d in G1.degree()]),
                'is_connected': nx.is_connected(G1),
                'density': nx.density(G1),
            },
            'graph2_properties': {
                'nodes': list(G2.nodes()),
                'edges': list(G2.edges()),
                'degree_sequence': sorted([d for n, d in G2.degree()]),
                'is_connected': nx.is_connected(G2),
                'density': nx.density(G2),
            }
        }
        
        with open(f"{base_path}_properties.json", 'w') as f:
            json.dump(properties, f, indent=2)
        
        # Save networkx graphs
        nx.write_gml(G1, f"{base_path}_graph1.gml")
        nx.write_gml(G2, f"{base_path}_graph2.gml")
        
        return safe_name
    
    def build_qubo(self, G1, G2, penalty_weight=20.0):
        """Builds the strict GI QUBO."""
        n = len(G1.nodes)
        qp = QuadraticProgram()
        
        # Variables x_{u,v}
        vars_map = {}
        for u in range(n):
            for v in range(n):
                name = f"x_{u}_{v}"
                qp.binary_var(name)
                vars_map[(u, v)] = name

        # Constraints: Row/Col sums = 1
        for i in range(n):
            qp.linear_constraint(linear={vars_map[(i, v)]: 1 for v in range(n)}, sense="==", rhs=1, name=f"row_{i}")
            qp.linear_constraint(linear={vars_map[(u, i)]: 1 for u in range(n)}, sense="==", rhs=1, name=f"col_{i}")

        # Objective: Edge mismatch penalty
        quad_dict = {}
        for u, up in product(range(n), repeat=2):
            if u == up: continue
            for v, vp in product(range(n), repeat=2):
                if v == vp: continue
                
                # If edge structure disagrees
                if G1.has_edge(u, up) != G2.has_edge(v, vp):
                    # Penalty for mapping u->v AND up->vp
                    key = (vars_map[(u, v)], vars_map[(up, vp)])
                    if key[0] > key[1]: key = (key[1], key[0])
                    quad_dict[key] = quad_dict.get(key, 0) + penalty_weight

        qp.minimize(quadratic=quad_dict)
        return qp, vars_map

    def solve_single_run(self, qp, reps=1, maxiter=100):
        """Runs QAOA once and captures the convergence trace."""
        history = []
        
        def callback(eval_count, params, mean, meta):
            history.append(mean)

        optimizer = COBYLA(maxiter=maxiter) 
        qaoa = QAOA(self.sampler, optimizer, reps=reps, callback=callback)
        
        solver = MinimumEigenOptimizer(qaoa)
        result = solver.solve(qp)
        
        return result, history

    def analyze_pair(self, G1, G2, name="Pair", runs=3, reps=1):
        """Analyze a single pair with FEWER runs (more diversity)."""
        print(f"\n{'='*60}")
        print(f"ANALYZING: {name} | Nodes: {len(G1)} | Qubits: {len(G1)**2}")
        print(f"{'='*60}")
        
        qp, vars_map = self.build_qubo(G1, G2)
        n = len(G1)
        
        best_result = None
        best_energy = float('inf')
        all_histories = []
        all_energies = []
        success_count = 0
        
        with tqdm(total=runs, desc=f"Quantum Runs for {name}") as pbar:
            for i in range(runs):
                result, hist = self.solve_single_run(qp, reps=reps)
                all_histories.append(hist)
                all_energies.append(result.fval)
                
                if result.fval < 1e-4:
                    success_count += 1
                
                if result.fval < best_energy:
                    best_energy = result.fval
                    best_result = result
                
                # Store in database
                self.results_db.append({
                    'pair_name': name,
                    'n_nodes': n,
                    'n_qubits': n**2,
                    'energy': result.fval,
                    'is_isomorphic': result.fval < 1e-4,
                    'run_id': i,
                    'convergence_steps': len(hist),
                    'final_energy': hist[-1] if hist else result.fval
                })
                
                pbar.update(1)
        
        # Store graph pair info
        self.graph_pairs.append({
            'name': name,
            'G1': G1.copy(),
            'G2': G2.copy(),
            'n_nodes': n,
            'energies': all_energies,
            'histories': all_histories,
            'best_energy': best_energy,
            'success_rate': success_count/runs
        })
        
        return best_result, all_histories, all_energies

    def generate_diverse_graph_pairs(self, n_nodes=4, num_pairs=10):
        """Generate MANY diverse graph pairs for analysis."""
        pairs = []
        
        # Generate isomorphic pairs
        isomorphic_types = [
            ("Complete vs Permuted", lambda: (nx.complete_graph(n_nodes), 
                                              nx.relabel_nodes(nx.complete_graph(n_nodes), 
                                                               {i: (i+1) % n_nodes for i in range(n_nodes)}))),
            ("Cycle vs Permuted", lambda: (nx.cycle_graph(n_nodes), 
                                           nx.relabel_nodes(nx.cycle_graph(n_nodes), 
                                                            {i: (i+2) % n_nodes for i in range(n_nodes)}))),
            ("Star vs Permuted", lambda: (nx.star_graph(n_nodes-1), 
                                          nx.relabel_nodes(nx.star_graph(n_nodes-1), 
                                                           {i: (i+1) % n_nodes for i in range(n_nodes)}))),
            ("Wheel vs Permuted", lambda: (nx.wheel_graph(n_nodes), 
                                           nx.relabel_nodes(nx.wheel_graph(n_nodes), 
                                                            {i: (i+1) % n_nodes for i in range(n_nodes)}))),
            ("Ladder vs Permuted", lambda: (nx.ladder_graph(n_nodes//2) if n_nodes % 2 == 0 and n_nodes >= 4 else None,
                                            nx.relabel_nodes(nx.ladder_graph(n_nodes//2), 
                                                             {i: (i+2) % n_nodes for i in range(n_nodes)}) if n_nodes % 2 == 0 and n_nodes >= 4 else None)),
            ("Path vs Permuted", lambda: (nx.path_graph(n_nodes), 
                                          nx.relabel_nodes(nx.path_graph(n_nodes), 
                                                           {i: (i+1) % n_nodes for i in range(n_nodes)}))),
            ("Regular vs Permuted", lambda: (nx.random_regular_graph(2, n_nodes, seed=42) if n_nodes >= 3 else None,
                                             nx.relabel_nodes(nx.random_regular_graph(2, n_nodes, seed=42), 
                                                              {i: (i+1) % n_nodes for i in range(n_nodes)}) if n_nodes >= 3 else None)),
            ("Bipartite vs Permuted", lambda: (nx.complete_bipartite_graph(n_nodes//2, n_nodes//2) if n_nodes % 2 == 0 else None,
                                               nx.relabel_nodes(nx.complete_bipartite_graph(n_nodes//2, n_nodes//2), 
                                                                {i: (i+1) % n_nodes for i in range(n_nodes)}) if n_nodes % 2 == 0 else None)),
        ]
        
        # Generate non-isomorphic pairs
        non_isomorphic_types = [
            ("Complete vs Cycle", lambda: (nx.complete_graph(n_nodes), nx.cycle_graph(n_nodes))),
            ("Cycle vs Path", lambda: (nx.cycle_graph(n_nodes), nx.path_graph(n_nodes))),
            ("Star vs Cycle", lambda: (nx.star_graph(n_nodes-1), nx.cycle_graph(n_nodes))),
            ("Complete vs Star", lambda: (nx.complete_graph(n_nodes), nx.star_graph(n_nodes-1))),
            ("Wheel vs Complete", lambda: (nx.wheel_graph(n_nodes), nx.complete_graph(n_nodes)) if n_nodes >= 4 else None),
            ("Random1 vs Random2", lambda: (nx.erdos_renyi_graph(n_nodes, 0.5, seed=42),
                                            nx.erdos_renyi_graph(n_nodes, 0.5, seed=43))),
            ("Regular vs Star", lambda: (nx.random_regular_graph(2, n_nodes, seed=42) if n_nodes >= 3 else None,
                                         nx.star_graph(n_nodes-1))),
            ("Ladder vs Cycle", lambda: (nx.ladder_graph(n_nodes//2) if n_nodes % 2 == 0 and n_nodes >= 4 else None,
                                         nx.cycle_graph(n_nodes))),
        ]
        
        # Select diverse pairs (mix of isomorphic and non-isomorphic)
        selected_pairs = []
        
        # Add isomorphic pairs
        iso_count = min(num_pairs // 2, len(isomorphic_types))
        for i in range(iso_count):
            name, generator = isomorphic_types[i]
            try:
                G1, G2 = generator()
                if G1 is not None and G2 is not None:
                    selected_pairs.append((f"{name}_{n_nodes}nodes", G1, G2))
            except:
                pass
        
        # Add non-isomorphic pairs
        noniso_count = min(num_pairs - iso_count, len(non_isomorphic_types))
        for i in range(noniso_count):
            name, generator = non_isomorphic_types[i]
            try:
                G1, G2 = generator()
                if G1 is not None and G2 is not None:
                    selected_pairs.append((f"{name}_{n_nodes}nodes", G1, G2))
            except:
                pass
        
        # Add some completely random pairs
        remaining = num_pairs - len(selected_pairs)
        for i in range(remaining):
            # Random isomorphic pair (same structure, random permutation)
            if np.random.random() > 0.5:
                # Generate random graph
                G1 = nx.erdos_renyi_graph(n_nodes, np.random.uniform(0.3, 0.7))
                # Make sure it's connected
                if not nx.is_connected(G1):
                    continue
                # Create isomorphic copy with random permutation
                mapping = {j: np.random.permutation(list(G1.nodes()))[j] for j in range(n_nodes)}
                G2 = nx.relabel_nodes(G1, mapping)
                selected_pairs.append((f"RandomIso_{i}_{n_nodes}nodes", G1, G2))
            else:
                # Random non-isomorphic pair (different structures)
                p1 = np.random.uniform(0.3, 0.7)
                p2 = np.random.uniform(0.3, 0.7)
                G1 = nx.erdos_renyi_graph(n_nodes, p1, seed=42+i*10)
                G2 = nx.erdos_renyi_graph(n_nodes, p2, seed=43+i*10)
                selected_pairs.append((f"RandomNonIso_{i}_{n_nodes}nodes", G1, G2))
        
        return selected_pairs

    def create_category_summary_plots(self):
        """Create summary plots organized by graph category."""
        if not self.results_db:
            print("No results available. Run analysis first.")
            return
        
        df = pd.DataFrame(self.results_db)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Energy distributions for isomorphic vs non-isomorphic
        ax1 = plt.subplot(3, 3, 1)
        iso_energies = df[df['is_isomorphic']]['energy']
        noniso_energies = df[~df['is_isomorphic']]['energy']
        
        if len(iso_energies) > 0:
            ax1.hist(iso_energies, bins=30, alpha=0.5, label='Isomorphic', color='green')
        if len(noniso_energies) > 0:
            ax1.hist(noniso_energies, bins=30, alpha=0.5, label='Non-Isomorphic', color='red')
        ax1.set_xlabel('Final Energy')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Energy Distribution by Isomorphism Class')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot comparison
        ax2 = plt.subplot(3, 3, 2)
        box_data = []
        box_labels = []
        if len(iso_energies) > 0:
            box_data.append(iso_energies)
            box_labels.append('Isomorphic')
        if len(noniso_energies) > 0:
            box_data.append(noniso_energies)
            box_labels.append('Non-Isomorphic')
        
        if box_data:
            ax2.boxplot(box_data, labels=box_labels)
        ax2.set_ylabel('Energy')
        ax2.set_title('Energy Distribution Comparison')
        ax2.grid(True, alpha=0.3)
        
        # 3. Success rate by graph size
        ax3 = plt.subplot(3, 3, 3)
        success_by_size = df.groupby(['n_nodes', 'is_isomorphic']).size().unstack()
        success_by_size.plot(kind='bar', ax=ax3)
        ax3.set_xlabel('Number of Nodes')
        ax3.set_ylabel('Count')
        ax3.set_title('Success Count by Graph Size')
        ax3.legend(['Non-Isomorphic', 'Isomorphic'])
        ax3.grid(True, alpha=0.3)
        
        # 4. Convergence speed comparison
        ax4 = plt.subplot(3, 3, 4)
        for pair in self.graph_pairs:
            color = 'green' if pair['success_rate'] > 0.8 else 'red'
            if pair['histories']:
                avg_history = np.mean([h for h in pair['histories'] if len(h) > 0], axis=0)
                if len(avg_history) > 0:
                    ax4.plot(avg_history, color=color, alpha=0.5, label=pair['name'])
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Energy')
        ax4.set_title('Average Convergence Traces')
        ax4.grid(True, alpha=0.3)
        
        # 5. Success rate scatter
        ax5 = plt.subplot(3, 3, 5)
        for pair in self.graph_pairs:
            x = pair['n_nodes']
            y = pair['success_rate']
            color = 'green' if y > 0.8 else 'red'
            ax5.scatter(x, y, color=color, s=100, alpha=0.6)
            ax5.annotate(pair['name'], (x, y), fontsize=8)
        ax5.set_xlabel('Number of Nodes')
        ax5.set_ylabel('Success Rate')
        ax5.set_title('Success Rate by Graph Complexity')
        ax5.grid(True, alpha=0.3)
        
        # 6. Energy statistics summary
        ax6 = plt.subplot(3, 3, 6)
        ax6.axis('off')
        summary_text = []
        
        if len(iso_energies) > 0:
            iso_stats = df[df['is_isomorphic']]['energy'].describe()
            summary_text.append("ISOMORPHIC PAIRS:")
            summary_text.append(f"Count: {len(iso_energies)}")
            summary_text.append(f"Mean Energy: {iso_stats['mean']:.4f}")
            summary_text.append(f"Std Dev: {iso_stats['std']:.4f}")
            summary_text.append(f"Min/Max: {iso_stats['min']:.4f}/{iso_stats['max']:.4f}")
        
        if len(noniso_energies) > 0:
            noniso_stats = df[~df['is_isomorphic']]['energy'].describe()
            summary_text.append("\nNON-ISOMORPHIC PAIRS:")
            summary_text.append(f"Count: {len(noniso_energies)}")
            summary_text.append(f"Mean Energy: {noniso_stats['mean']:.4f}")
            summary_text.append(f"Std Dev: {noniso_stats['std']:.4f}")
            summary_text.append(f"Min/Max: {noniso_stats['min']:.4f}/{noniso_stats['max']:.4f}")
        
        if len(iso_energies) > 0 and len(noniso_energies) > 0:
            t_test = stats.ttest_ind(iso_energies, noniso_energies, equal_var=False)
            summary_text.append(f"\nT-test p-value: {t_test.pvalue:.6f}")
        
        ax6.text(0.1, 0.9, "\n".join(summary_text), fontsize=10, 
                family='monospace', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 7. Heatmap of energy distributions
        ax7 = plt.subplot(3, 3, 7)
        energy_matrix = []
        pair_names = []
        for pair in self.graph_pairs:
            if pair['energies']:
                energy_matrix.append(pair['energies'])
                pair_names.append(pair['name'])
        
        if energy_matrix:
            # Pad with NaN if lengths differ
            max_len = max(len(e) for e in energy_matrix)
            padded_matrix = []
            for e in energy_matrix:
                padded = list(e) + [np.nan] * (max_len - len(e))
                padded_matrix.append(padded)
            
            im = ax7.imshow(padded_matrix, aspect='auto', cmap='viridis')
            ax7.set_yticks(range(len(pair_names)))
            ax7.set_yticklabels(pair_names, fontsize=8)
            ax7.set_xlabel('Run Number')
            ax7.set_title('Energy Heatmap Across Runs')
            plt.colorbar(im, ax=ax7)
        
        # 8. ROC-style analysis
        ax8 = plt.subplot(3, 3, 8)
        if len(df) > 0:
            thresholds = np.linspace(df['energy'].min(), df['energy'].max(), 100)
            true_positive_rates = []
            false_positive_rates = []
            
            for threshold in thresholds:
                pred_iso = df['energy'] < threshold
                true_iso = df['is_isomorphic']
                
                tp = np.sum(pred_iso & true_iso)
                fp = np.sum(pred_iso & ~true_iso)
                tn = np.sum(~pred_iso & ~true_iso)
                fn = np.sum(~pred_iso & true_iso)
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                true_positive_rates.append(tpr)
                false_positive_rates.append(fpr)
            
            ax8.plot(false_positive_rates, true_positive_rates, 'b-', linewidth=2)
            ax8.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax8.set_xlabel('False Positive Rate')
            ax8.set_ylabel('True Positive Rate')
            ax8.set_title('Discrimination Performance')
            ax8.grid(True, alpha=0.3)
            ax8.set_aspect('equal')
        
        # 9. Feature space visualization (PCA)
        ax9 = plt.subplot(3, 3, 9)
        if len(df) > 10:
            features = pd.DataFrame({
                'energy': df['energy'],
                'convergence_steps': df['convergence_steps'],
                'n_nodes': df['n_nodes'],
                'n_qubits': df['n_qubits']
            })
            
            pca = PCA(n_components=2)
            try:
                pca_result = pca.fit_transform(features)
                colors = ['green' if iso else 'red' for iso in df['is_isomorphic']]
                ax9.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, alpha=0.6)
                ax9.set_xlabel('PCA Component 1')
                ax9.set_ylabel('PCA Component 2')
                ax9.set_title('Feature Space (PCA)')
                ax9.grid(True, alpha=0.3)
            except Exception as e:
                ax9.text(0.5, 0.5, f'PCA failed: {str(e)[:30]}', ha='center', va='center')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistical summary
        print("\n" + "="*80)
        print("DISTINGUISHABILITY ANALYSIS SUMMARY")
        print("="*80)
        print(f"Total runs: {len(df)}")
        print(f"Total unique graph pairs: {len(self.graph_pairs)}")
        if len(iso_energies) > 0:
            print(f"Isomorphic pairs: {len(iso_energies)} runs ({df['is_isomorphic'].sum()} pairs)")
        if len(noniso_energies) > 0:
            print(f"Non-isomorphic pairs: {len(noniso_energies)} runs ({len(df) - df['is_isomorphic'].sum()} pairs)")
        
        if len(df) > 0:
            overall_accuracy = (df['is_isomorphic'] == (df['energy'] < 1e-4)).mean()
            print(f"Overall accuracy: {overall_accuracy:.2%}")
            
            if len(iso_energies) > 0 and len(noniso_energies) > 0:
                mean_diff = noniso_stats['mean'] - iso_stats['mean']
                print(f"Mean energy difference: {mean_diff:.4f}")
                print(f"T-test p-value: {t_test.pvalue:.6f}")
        print("="*80)

# %%
# ==========================================
# MAIN EXECUTION - MORE DIVERSITY, FEWER RUNS
# ==========================================

if __name__ == "__main__":
    # Setup output directories
    print("Setting up output directories...")
    output_dirs = setup_output_directories()
    print(f"Results will be saved to: {output_dirs['base']}")
    
    # Initialize analyzer
    analyzer = GraphIsoAnalyzer(output_dirs)
    
    # Generate MANY diverse graph pairs (more diversity, fewer runs per pair)
    print("\nGenerating DIVERSE graph pairs for analysis...")
    
    # Parameters for diversity
    MAX_NODES = 5  # Maximum graph size
    PAIRS_PER_SIZE = 30  # More pairs per size
    RUNS_PER_PAIR = 3   # Fewer runs per pair
    
    all_pairs = []
    
    # Generate pairs for each graph size
    for n_nodes in range(2, MAX_NODES + 1):  # Start from 2 nodes
        print(f"\nGenerating {PAIRS_PER_SIZE} diverse pairs with {n_nodes} nodes...")
        
        # Generate diverse pairs for this size
        size_pairs = analyzer.generate_diverse_graph_pairs(n_nodes=n_nodes, num_pairs=PAIRS_PER_SIZE)
        all_pairs.extend(size_pairs)
        
        print(f"  Added {len(size_pairs)} pairs")
    
    # Add single node case
    G_single = nx.Graph()
    G_single.add_node(0)
    all_pairs.append(("Single Node", G_single, G_single.copy()))
    
    # Add some special small world networks (4 nodes)
    if MAX_NODES >= 4:
        G_sw1 = nx.watts_strogatz_graph(4, 2, 0.3, seed=42)
        G_sw2 = nx.watts_strogatz_graph(4, 2, 0.3, seed=43)
        all_pairs.append(("Small World Pair (4 nodes)", G_sw1, G_sw2))
    
    # Add bipartite graphs (4 nodes)
    if MAX_NODES >= 4:
        bipartite1 = nx.complete_bipartite_graph(2, 2)
        mapping = {i: (i+1) % 4 for i in range(4)}
        bipartite2 = nx.relabel_nodes(bipartite1, mapping)
        all_pairs.append(("Bipartite vs Bipartite (4 nodes)", bipartite1, bipartite2))
    
    # Filter out any None graphs
    all_pairs = [(name, G1, G2) for name, G1, G2 in all_pairs if G1 is not None and G2 is not None]
    
    # Filter out any pairs with more than MAX_NODES nodes
    all_pairs = [(name, G1, G2) for name, G1, G2 in all_pairs if len(G1.nodes) <= MAX_NODES]
    
    # Run analysis with FEWER runs per pair
    print(f"\nTotal pairs to analyze: {len(all_pairs)}")
    print(f"Runs per pair: {RUNS_PER_PAIR}")
    print(f"Expected total runs: {len(all_pairs) * RUNS_PER_PAIR}")
    
    # Process all pairs
    for pair_name, G1, G2 in tqdm(all_pairs, desc="Analyzing diverse graph pairs"):
        analyzer.analyze_pair(G1, G2, name=pair_name, runs=RUNS_PER_PAIR)
    
    # Create summary plots
    print("\nCreating summary plots...")
    analyzer.create_category_summary_plots()
    
    # Save metadata and final results
    analyzer.save_metadata()
    
    # Save all results to pickle for later analysis
    results_file = os.path.join(output_dirs['data'], 'all_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump({
            'results_db': analyzer.results_db,
            'graph_pairs': analyzer.graph_pairs,
            'output_dirs': output_dirs,
            'experiment_config': {
                'max_nodes': MAX_NODES,
                'pairs_per_size': PAIRS_PER_SIZE,
                'runs_per_pair': RUNS_PER_PAIR,
                'total_pairs': len(all_pairs)
            }
        }, f)
    
    # Final summary
    print("\n" + "="*80)
    print("DIVERSE GRAPH ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dirs['base']}")
    
    # Quick statistics
    df = pd.DataFrame(analyzer.results_db)
    if len(df) > 0:
        print(f"\nExperiment Configuration:")
        print(f"Maximum nodes: {MAX_NODES}")
        print(f"Pairs per size: {PAIRS_PER_SIZE}")
        print(f"Runs per pair: {RUNS_PER_PAIR}")
        
        print(f"\nPerformance Summary:")
        print(f"Total quantum runs: {len(df)}")
        print(f"Total unique graph pairs: {len(df['pair_name'].unique())}")
        print(f"Graph sizes analyzed: {sorted(df['n_nodes'].unique())}")
        
        iso_mask = df['is_isomorphic'] == True
        if iso_mask.any():
            print(f"Isomorphic pairs: {iso_mask.sum()} runs ({iso_mask.sum()/RUNS_PER_PAIR:.0f} unique pairs)")
            print(f"  Mean energy: {df[iso_mask]['energy'].mean():.6f}")
        
        if (~iso_mask).any():
            print(f"Non-isomorphic pairs: {(~iso_mask).sum()} runs ({(~iso_mask).sum()/RUNS_PER_PAIR:.0f} unique pairs)")
            print(f"  Mean energy: {df[~iso_mask]['energy'].mean():.6f}")
        
        # Calculate overall accuracy
        threshold = 1e-4
        predictions = df['energy'] < threshold
        accuracy = (predictions == df['is_isomorphic']).mean()
        print(f"\nOverall discrimination accuracy: {accuracy:.2%}")
        
        # Confusion matrix
        true_positive = ((predictions == True) & (df['is_isomorphic'] == True)).sum()
        false_positive = ((predictions == True) & (df['is_isomorphic'] == False)).sum()
        true_negative = ((predictions == False) & (df['is_isomorphic'] == False)).sum()
        false_negative = ((predictions == False) & (df['is_isomorphic'] == True)).sum()
        
        print(f"\nConfusion Matrix:")
        print(f"True Positives (correct isomorphic): {true_positive}")
        print(f"False Positives (wrong isomorphic): {false_positive}")
        print(f"True Negatives (correct non-isomorphic): {true_negative}")
        print(f"False Negatives (wrong non-isomorphic): {false_negative}")
        
        # Performance metrics
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nPerformance Metrics:")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"F1 Score: {f1_score:.2%}")
        
        # Diversity metrics
        print(f"\nDiversity Analysis:")
        pair_categories = {}
        for pair in analyzer.graph_pairs:
            category = pair['name'].split('_')[0]  # Get category from name
            if category not in pair_categories:
                pair_categories[category] = []
            pair_categories[category].append(pair)
        
        print(f"Unique graph categories: {len(pair_categories)}")
        print("Categories sampled:")
        for category, pairs in sorted(pair_categories.items()):
            iso_count = sum(1 for p in pairs if p['best_energy'] < 1e-4)
            noniso_count = len(pairs) - iso_count
            print(f"  {category}: {len(pairs)} pairs ({iso_count} isomorphic, {noniso_count} non-isomorphic)")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nSummary:")
    print(f"• Analyzed {len(analyzer.graph_pairs)} diverse graph pairs")
    print(f"• Each pair was run {RUNS_PER_PAIR} times")
    print(f"• Total computational effort: {len(df)} quantum runs")
    print(f"• Graph sizes: 1 to {MAX_NODES} nodes")
    print(f"• Results saved in organized folders with 300 DPI figures")

# %%