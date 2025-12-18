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

    def analyze_pair(self, G1, G2, name="Pair", runs=20, reps=1):
        """Analyze a single pair with multiple runs and save results."""
        print(f"\n{'='*60}")
        print(f"ANALYZING: {name} | Nodes: {len(G1)} | Qubits: {len(G1)**2}")
        print(f"{'='*60}")
        
        # Save graph data first
        safe_name = self.save_graph_data(G1, G2, name)
        
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
                    'safe_name': safe_name,
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
            'safe_name': safe_name,
            'G1': G1.copy(),
            'G2': G2.copy(),
            'n_nodes': n,
            'energies': all_energies,
            'histories': all_histories,
            'best_energy': best_energy,
            'success_rate': success_count/runs
        })
        
        # Save analysis results for this pair
        self.save_pair_results(name, safe_name, all_energies, all_histories, best_result)
        
        return best_result, all_histories, all_energies
    
    def save_pair_results(self, pair_name, safe_name, energies, histories, best_result):
        """Save detailed results for a specific pair."""
        base_path = os.path.join(self.output_dirs['data'], safe_name)
        
        # Save energies
        np.save(f"{base_path}_energies.npy", energies)
        
        # Save convergence histories
        with open(f"{base_path}_histories.pkl", 'wb') as f:
            pickle.dump(histories, f)
        
        # Save best result
        if best_result is not None:
            result_data = {
                'fval': best_result.fval,
                'x': best_result.x.tolist() if hasattr(best_result.x, 'tolist') else list(best_result.x),
                'status': str(best_result.status)
            }
            with open(f"{base_path}_best_result.json", 'w') as f:
                json.dump(result_data, f, indent=2)
        
        # Create energy distribution plot
        plt.figure(figsize=(8, 6), dpi=300)
        plt.hist(energies, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        plt.axvline(x=1e-4, color='red', linestyle='--', linewidth=2, label='Isomorphism Threshold')
        plt.xlabel('Final Energy', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Energy Distribution: {pair_name}\n(n={len(energies)} runs)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dirs['energy_distributions'], f"{safe_name}_energy_dist.png"), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Create convergence plot
        plt.figure(figsize=(10, 6), dpi=300)
        for i, hist in enumerate(histories):
            if len(hist) > 0:
                plt.plot(hist, alpha=0.3, color='blue', linewidth=0.5)
        
        # Plot average convergence
        if histories and any(len(h) > 0 for h in histories):
            valid_histories = [h for h in histories if len(h) > 0]
            min_len = min(len(h) for h in valid_histories)
            if min_len > 0:
                avg_hist = np.mean([h[:min_len] for h in valid_histories], axis=0)
                plt.plot(avg_hist, color='red', linewidth=2, label='Average')
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Energy', fontsize=12)
        plt.title(f'Convergence Traces: {pair_name}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dirs['convergence_plots'], f"{safe_name}_convergence.png"), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Save mapping matrix if best result exists
        if best_result is not None and hasattr(best_result, 'x'):
            n = int(np.sqrt(len(best_result.x)))  # Assuming square matrix
            mapping_matrix = best_result.x.reshape(n, n)
            
            plt.figure(figsize=(6, 5), dpi=300)
            sns.heatmap(mapping_matrix, annot=True, cmap="YlOrRd", cbar=True, 
                       linewidths=0.5, linecolor='gray', fmt='.1f')
            plt.xlabel('Graph 2 Nodes', fontsize=12)
            plt.ylabel('Graph 1 Nodes', fontsize=12)
            plt.title(f'Best Mapping Matrix: {pair_name}\nEnergy: {best_result.fval:.4f}', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dirs['mapping_matrices'], f"{safe_name}_mapping.png"), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

    def generate_graph_pairs(self, n_nodes=4):
        """Generate various graph pairs for analysis."""
        pairs = []
        
        # 1. Complete graph vs itself (trivially isomorphic)
        K_n = nx.complete_graph(n_nodes)
        pairs.append(("Complete vs Complete", K_n, K_n.copy()))
        
        # 2. Complete graph vs random permutation
        K_n_perm = nx.relabel_nodes(K_n, {i: (i+1) % n_nodes for i in range(n_nodes)})
        pairs.append(("Complete vs Permuted", K_n, K_n_perm))
        
        # 3. Cycle vs itself
        C_n = nx.cycle_graph(n_nodes)
        pairs.append(("Cycle vs Cycle", C_n, C_n.copy()))
        
        # 4. Cycle vs path (non-isomorphic)
        P_n = nx.path_graph(n_nodes)
        pairs.append(("Cycle vs Path", C_n, P_n))
        
        # 5. Star vs itself
        S_n = nx.star_graph(n_nodes-1)
        pairs.append(("Star vs Star", S_n, S_n.copy()))
        
        # 6. Star vs different structure (non-isomorphic)
        if n_nodes >= 4:
            G_random = nx.erdos_renyi_graph(n_nodes, 0.5)
            pairs.append(("Star vs Random", S_n, G_random))
        
        # 7. Two different random graphs (likely non-isomorphic)
        G1 = nx.erdos_renyi_graph(n_nodes, 0.5, seed=42)
        G2 = nx.erdos_renyi_graph(n_nodes, 0.5, seed=43)
        pairs.append(("Random vs Random", G1, G2))
        
        # 8. Regular graph vs itself
        if n_nodes % 2 == 0:
            R_n = nx.random_regular_graph(2, n_nodes, seed=42)
            pairs.append(("Regular vs Regular", R_n, R_n.copy()))
        
        # 9. Ladder graph vs itself
        if n_nodes >= 4 and n_nodes % 2 == 0:
            L_n = nx.ladder_graph(n_nodes // 2)
            # Permute ladder graph
            mapping = {i: (i+2) % n_nodes for i in range(n_nodes)}
            L_n_perm = nx.relabel_nodes(L_n, mapping)
            pairs.append(("Ladder vs Ladder", L_n, L_n_perm))
        
        # 10. Wheel graph vs itself (for n_nodes >= 4)
        if n_nodes >= 4:
            W_n = nx.wheel_graph(n_nodes)
            # Permute wheel graph
            mapping = {i: (i+1) % n_nodes for i in range(n_nodes)}
            W_n_perm = nx.relabel_nodes(W_n, mapping)
            pairs.append(("Wheel vs Wheel", W_n, W_n_perm))
        
        # 11. Barbell graph vs itself (for even n_nodes >= 6)
        if n_nodes >= 6 and n_nodes % 2 == 0:
            B_n = nx.barbell_graph(n_nodes//2 - 1, 2)
            if len(B_n.nodes()) == n_nodes:  # Ensure correct size
                mapping = {i: (i+1) % n_nodes for i in range(n_nodes)}
                B_n_perm = nx.relabel_nodes(B_n, mapping)
                pairs.append(("Barbell vs Barbell", B_n, B_n_perm))
        
        return pairs

    def create_category_summary_plots(self):
        """Create summary plots organized by graph category."""
        if not self.results_db:
            print("No results available. Run analysis first.")
            return
        
        df = pd.DataFrame(self.results_db)
        
        # 1. Summary by graph size
        plt.figure(figsize=(10, 6), dpi=300)
        success_by_size = df.groupby('n_nodes').apply(
            lambda x: (x['energy'] < 1e-4).mean()
        ).reset_index()
        success_by_size.columns = ['n_nodes', 'success_rate']
        
        plt.bar(success_by_size['n_nodes'].astype(str), success_by_size['success_rate'], 
                color='steelblue', alpha=0.7, edgecolor='black')
        plt.xlabel('Number of Nodes', fontsize=12)
        plt.ylabel('Success Rate', fontsize=12)
        plt.title('Success Rate by Graph Size', fontsize=14, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(success_by_size['success_rate']):
            plt.text(i, v + 0.02, f'{v:.2%}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dirs['summary_plots'], 'success_by_size.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # 2. Energy distribution by category
        categories = {}
        for pair in self.graph_pairs:
            category = pair['name'].split(' vs ')[0]  # Get first part as category
            if category not in categories:
                categories[category] = []
            categories[category].extend(pair['energies'])
        
        if categories:
            fig, axes = plt.subplots(len(categories), 1, figsize=(10, 4*len(categories)), dpi=300)
            if len(categories) == 1:
                axes = [axes]
            
            for ax, (category, energies) in zip(axes, categories.items()):
                ax.hist(energies, bins=20, alpha=0.7, color='teal', edgecolor='black')
                ax.axvline(x=1e-4, color='red', linestyle='--', linewidth=2)
                ax.set_xlabel('Energy', fontsize=11)
                ax.set_ylabel('Frequency', fontsize=11)
                ax.set_title(f'{category} Graphs\n(n={len(energies)} runs)', fontsize=12)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dirs['summary_plots'], 'energy_by_category.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        # 3. Performance comparison: Isomorphic vs Non-isomorphic
        iso_mask = df['is_isomorphic'] == True
        if iso_mask.any() and (~iso_mask).any():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
            
            # Box plot
            iso_energies = df[iso_mask]['energy']
            noniso_energies = df[~iso_mask]['energy']
            
            ax1.boxplot([iso_energies, noniso_energies], 
                       labels=['Isomorphic', 'Non-Isomorphic'])
            ax1.set_ylabel('Energy', fontsize=12)
            ax1.set_title('Energy Distribution Comparison', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Violin plot
            data = [iso_energies, noniso_energies]
            parts = ax2.violinplot(data, showmeans=True, showmedians=True)
            ax2.set_xticks([1, 2])
            ax2.set_xticklabels(['Isomorphic', 'Non-Isomorphic'])
            ax2.set_ylabel('Energy', fontsize=12)
            ax2.set_title('Energy Density Comparison', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Color the violins
            for pc, color in zip(parts['bodies'], ['lightgreen', 'lightcoral']):
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dirs['summary_plots'], 'performance_comparison.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        # 4. Save comprehensive summary table
        summary_data = []
        for pair in self.graph_pairs:
            summary_data.append({
                'Pair Name': pair['name'],
                'Nodes': pair['n_nodes'],
                'Qubits': pair['n_nodes']**2,
                'Success Rate': f"{pair['success_rate']:.2%}",
                'Best Energy': f"{pair['best_energy']:.6f}",
                'Mean Energy': f"{np.mean(pair['energies']):.6f}",
                'Std Energy': f"{np.std(pair['energies']):.6f}",
                'Is Isomorphic': 'Yes' if pair['best_energy'] < 1e-4 else 'No'
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(self.output_dirs['summary_plots'], 'summary_table.csv'), index=False)
        summary_df.to_latex(os.path.join(self.output_dirs['summary_plots'], 'summary_table.tex'), index=False)
        
        # 5. Create README file
        readme_content = f"""# Graph Isomorphism Analysis Results

## Experiment Details
- Start Time: {self.experiment_metadata['start_time']}
- Total Runs: {len(self.results_db)}
- Unique Graph Pairs: {len(self.graph_pairs)}
- Graph Sizes: {sorted(set(df['n_nodes']))}

## Directory Structure
- `graph_pairs/`: Visualizations of graph pairs and adjacency matrices
- `energy_distributions/`: Energy histograms for each pair
- `convergence_plots/`: Convergence traces for each pair
- `mapping_matrices/`: Best found mapping matrices
- `summary_plots/`: Aggregated results and statistics
- `data/`: Raw data files (JSON, NPY, PKL)

## Analysis Summary
{summary_df.to_string()}

## Methodology
- Solver: QAOA with COBYLA optimizer
- Sampler: StatevectorSampler
- Repetitions: 1 QAOA layer
- Threshold for isomorphism: Energy < 1e-4
"""
        
        with open(os.path.join(self.output_dirs['base'], 'README.md'), 'w') as f:
            f.write(readme_content)

# %%
# ==========================================
# MAIN EXECUTION
# ==========================================



if __name__ == "__main__":
    # Setup output directories
    print("Setting up output directories...")
    output_dirs = setup_output_directories()
    print(f"Results will be saved to: {output_dirs['base']}")
    
    # Initialize analyzer
    analyzer = GraphIsoAnalyzer(output_dirs)
    
    # Generate diverse graph pairs (ONLY up to 4 nodes)
    print("\nGenerating graph pairs for analysis...")
    graph_pairs_3 = analyzer.generate_graph_pairs(n_nodes=3)
    graph_pairs_4 = analyzer.generate_graph_pairs(n_nodes=4)
    # REMOVE or COMMENT OUT the 5-node generation
    # graph_pairs_5 = analyzer.generate_graph_pairs(n_nodes=5)
    
    # Combine all pairs (only 3 and 4 nodes)
    all_pairs = graph_pairs_3 + graph_pairs_4
    # Remove the 5-node pairs from the combination
    
    # Add some special cases (only up to 4 nodes)
    # Trivial case: Single node
    G_single = nx.Graph()
    G_single.add_node(0)
    all_pairs.append(("Single Node", G_single, G_single.copy()))
    
    # Complex case: Small world networks (4 nodes max)
    # Change from 6 to 4 nodes
    G_sw1 = nx.watts_strogatz_graph(4, 2, 0.3, seed=42)  # Changed from (6, 3)
    G_sw2 = nx.watts_strogatz_graph(4, 2, 0.3, seed=43)  # Changed from (6, 3)
    all_pairs.append(("Small World Pair (4 nodes)", G_sw1, G_sw2))
    
    # Add bipartite graphs (4 nodes max)
    # Change from (3, 3) to (2, 2) for 4 nodes total
    bipartite1 = nx.complete_bipartite_graph(2, 2)  # Changed from (3, 3)
    mapping = {i: (i+1) % 4 for i in range(4)}  # Changed from 6 to 4
    bipartite2 = nx.relabel_nodes(bipartite1, mapping)
    all_pairs.append(("Bipartite vs Bipartite (4 nodes)", bipartite1, bipartite2))
    
    # Filter out any pairs with more than 4 nodes (just to be safe)
    all_pairs = [(name, G1, G2) for name, G1, G2 in all_pairs if len(G1.nodes) <= 4]
    
    # Run analysis with different runs per size to manage computation time
    print(f"\nTotal pairs to analyze: {len(all_pairs)}")
    
    # Configure runs based on graph size (only for sizes we have)
    runs_by_size = {
        1: 20,   # 1 node: 20 runs
        2: 20,   # 2 nodes: 20 runs  
        3: 15,   # 3 nodes: 15 runs
        4: 10,   # 4 nodes: 10 runs
        # REMOVED: 5: 8,    # 5 nodes: 8 runs
        # REMOVED: 6: 5     # 6 nodes: 5 runs
    }
    
    # Process pairs by size
    for n_size in sorted(set(len(G1.nodes) for _, G1, _ in all_pairs)):
        if n_size not in runs_by_size:
            print(f"Skipping {n_size}-node graphs (not in runs_by_size configuration)")
            continue
            
        size_pairs = [(name, G1, G2) for name, G1, G2 in all_pairs 
                     if len(G1.nodes) == n_size]
        runs = runs_by_size.get(n_size, 5)
        
        if size_pairs:
            print(f"\nAnalyzing {n_size}-node graphs ({len(size_pairs)} pairs, {runs} runs each)...")
            for pair_name, G1, G2 in size_pairs:
                analyzer.analyze_pair(G1, G2, name=pair_name, runs=runs)
    
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
            'output_dirs': output_dirs
        }, f)
    
    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dirs['base']}")
    
    # Quick statistics
    df = pd.DataFrame(analyzer.results_db)
    if len(df) > 0:
        print(f"\nQuick Statistics:")
        print(f"Total quantum runs: {len(df)}")
        print(f"Unique graph pairs: {len(df['pair_name'].unique())}")
        print(f"Graph sizes analyzed: {sorted(df['n_nodes'].unique())}")
        
        iso_mask = df['is_isomorphic'] == True
        if iso_mask.any():
            print(f"Isomorphic pairs: {iso_mask.sum()} runs")
            print(f"  Mean energy: {df[iso_mask]['energy'].mean():.6f}")
        
        if (~iso_mask).any():
            print(f"Non-isomorphic pairs: {(~iso_mask).sum()} runs")
            print(f"  Mean energy: {df[~iso_mask]['energy'].mean():.6f}")
        
        # Calculate overall accuracy
        threshold = 1e-4
        predictions = df['energy'] < threshold
        accuracy = (predictions == df['is_isomorphic']).mean()
        print(f"\nOverall discrimination accuracy: {accuracy:.2%}")
    
    print("\nCheck the README.md file in the results directory for detailed analysis.")
    print("="*80)

# %%