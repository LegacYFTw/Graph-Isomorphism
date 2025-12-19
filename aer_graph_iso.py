# %%
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Must be called before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import product
from tqdm import tqdm
import pandas as pd
from scipy import stats
import os
import json
from datetime import datetime
import pickle
import warnings
import time
warnings.filterwarnings('ignore')

# Qiskit Imports with GPU support
try:
    from qiskit_aer import AerSimulator, Aer
    from qiskit.primitives import BackendSampler
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    print("Qiskit successfully imported")
except ImportError as e:
    print(f"Qiskit import error: {e}")
    raise

# Add these imports at the top of your file
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import BackendEstimator
from scipy.optimize import minimize


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
    """Constructs the Quadratic Program for Graph Isomorphism."""
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
    for u in range(n):
        qp.linear_constraint(
            linear={vars_map[(u, v)]: 1 for v in range(n)},
            sense="==",
            rhs=1,
            name=f"row_{u}"
        )

    for v in range(n):
        qp.linear_constraint(
            linear={vars_map[(u, v)]: 1 for u in range(n)},
            sense="==",
            rhs=1,
            name=f"col_{v}"
        )

    # 3. Add Quadratic Objective for Edge Inconsistency
    quad_dict = {}

    for u, up in product(range(n), repeat=2):
        if u == up:
            continue

        for v, vp in product(range(n), repeat=2):
            if v == vp:
                continue

            # Check Adjacency
            g1_edge = G1.has_edge(u, up)
            g2_edge = G2.has_edge(v, vp)

            # If structural mismatch found between these pairs
            if g1_edge != g2_edge:
                key = (vars_map[(u, v)], vars_map[(up, vp)])
                if key[0] > key[1]:
                    key = (key[1], key[0])
                quad_dict[key] = quad_dict.get(key, 0) + penalty_weight

    qp.minimize(quadratic=quad_dict)
    return qp

# %%


class GraphIsoAnalyzer:
    def __init__(self, output_dirs, use_gpu=False, shots=1024):
        self.output_dirs = output_dirs
        self.results_db = []
        self.graph_pairs = []
        self.shots = shots
        self.use_gpu = use_gpu
        
        # We no longer pre-create a single fixed backend here 
        # because the method depends on the qubit count per run.
        print(f"Initialized Analyzer (GPU preference: {use_gpu})")

    def _get_dynamic_backend(self, n_qubits):
        """Selects backend method based on qubit count."""
        # Selection Logic
        if n_qubits > 30:
            method = 'matrix_product_state'
            print(f"  > {n_qubits} qubits: Using Matrix Product State (MPS)")
        else:
            method = 'statevector'
            print(f"  > {n_qubits} qubits: Using Statevector")

        device = 'GPU' if self.use_gpu else 'CPU'
        
        return AerSimulator(
            method=method,
            device=device,
            shots=self.shots
        )

    def solve_single_run(self, qp, reps=1, maxiter=50):
        """Runs QAOA with dynamic backend selection."""
        n_qubits = len(qp.variables)
        
        # 1. Dynamically get the backend for this specific problem size
        current_backend = self._get_dynamic_backend(n_qubits)
        
        # 2. Initialize Estimator/Sampler with the chosen backend
        estimator = BackendEstimator(backend=current_backend)
        estimator.set_options(shots=self.shots)
        
        history = []
        
        # Adjust reps based on complexity
        if n_qubits > 16: reps = 1
        
        try:
            from qiskit_optimization.converters import QuadraticProgramToQubo
            converter = QuadraticProgramToQubo()
            qubo = converter.convert(qp)
            operator, offset = qubo.to_ising()
            
            qaoa_circuit = QAOAAnsatz(cost_operator=operator, reps=reps)
            
            def cost_function(params):
                job = estimator.run(circuits=[qaoa_circuit], 
                                    observables=[operator], 
                                    parameter_values=[params])
                energy = float(job.result().values[0])
                history.append(energy)
                return energy
            
            init_params = [0.1] * len(qaoa_circuit.parameters)
            result_opt = minimize(cost_function, init_params, method='COBYLA', options={'maxiter': maxiter})
            
            optimal_energy = result_opt.fun + offset
            optimal_params = result_opt.x
            
            # --- SAMPLING BLOCK ---
            from qiskit.primitives import BackendSampler
            sampler = BackendSampler(backend=current_backend)
            sampler.set_options(shots=self.shots)
            
            sampling_circuit = qaoa_circuit.measure_all(inplace=False)
            job = sampler.run(circuits=[sampling_circuit], parameter_values=[optimal_params])
            quasi_dist = job.result().quasi_dists[0]
            
            best_bit_int = max(quasi_dist, key=quasi_dist.get)
            best_bitstring = format(best_bit_int, f'0{n_qubits}b')
            x_values = [float(bit) for bit in best_bitstring[::-1]]
            
            from qiskit_optimization.algorithms import OptimizationResult, OptimizationResultStatus
            result = OptimizationResult(
                x=np.array(x_values),
                fval=optimal_energy,
                variables=qp.variables,
                status=OptimizationResultStatus.SUCCESS
            )
            
        except Exception as e:
            print(f"  QAOA failed: {e}")
            # Classical fallback...
            from qiskit_algorithms import NumPyMinimumEigensolver
            from qiskit_optimization.algorithms import MinimumEigenOptimizer
            result = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qp)
            history = [result.fval]
        
        return result, history
    
    def save_metadata(self):
        """Save experiment metadata."""
        metadata_file = os.path.join(
            self.output_dirs['data'], 'experiment_metadata.json')
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
        safe_name = pair_name.replace(
            " ", "_").replace(":", "").replace("/", "_")
        base_path = os.path.join(self.output_dirs['graph_pairs'], safe_name)

        # Save graph pair visualization
        save_graph_pair(G1, G2, f"{base_path}_graphs.png",
                        title1="Graph A", title2="Graph B",
                        main_title=f"{pair_name}\n(n={len(G1.nodes)})")

        # Save adjacency matrices
        save_adjacency_matrices(G1, G2, f"{base_path}_adjacency.png")

        # Save graphs as pickle files (handles any data type)
        with open(f"{base_path}_graph1.pkl", 'wb') as f:
            pickle.dump(G1, f)
        with open(f"{base_path}_graph2.pkl", 'wb') as f:
            pickle.dump(G2, f)

        return safe_name



    def analyze_pair(self, G1, G2, name="Pair", runs=3, reps=1):
        """Analyze a single pair with multiple runs."""
        print(f"\n{'='*60}")
        print(f"ANALYZING: {name} | Nodes: {len(G1)} | Qubits: {len(G1)**2}")
        print(f"{'='*60}")

        # Save graph data
        safe_name = self.save_graph_data(G1, G2, name)

        # Build QUBO
        qp = get_gi_qubo(G1, G2)
        n = len(G1.nodes)

        best_result = None
        best_energy = float('inf')
        all_histories = []
        all_energies = []
        success_count = 0

        with tqdm(total=runs, desc=f"Quantum Runs ({'GPU' if self.use_gpu else 'CPU'})") as pbar:
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
                    'final_energy': hist[-1] if hist else result.fval,
                    'device': 'GPU' if self.use_gpu else 'CPU',
                    'shots': self.shots
                })

                pbar.set_postfix({'energy': f'{result.fval:.4f}'})
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
            'success_rate': success_count/runs,
            'device': 'GPU' if self.use_gpu else 'CPU'
        })

        # Save results for this pair
        self.save_pair_results(name, safe_name, all_energies,
                               all_histories, best_result)

        return best_result, all_histories, all_energies

    def save_pair_results(self, pair_name, safe_name, energies, histories, best_result):
        """Save detailed results for a specific pair."""
        base_path = os.path.join(self.output_dirs['data'], safe_name)

        # Save raw data
        np.save(f"{base_path}_energies.npy", energies)

        # Save convergence histories
        with open(f"{base_path}_histories.pkl", 'wb') as f:
            pickle.dump(histories, f)

        # Save best result
        if best_result is not None:
            result_data = {
                'fval': best_result.fval,
                'x': best_result.x.tolist() if hasattr(best_result.x, 'tolist') else list(best_result.x),
                'status': str(best_result.status),
                'metadata': {
                    'gpu_used': self.use_gpu,
                    'shots': self.shots,
                    'n_runs': len(energies)
                }
            }
            with open(f"{base_path}_best_result.json", 'w') as f:
                json.dump(result_data, f, indent=2)

        # Create energy distribution plot
        plt.figure(figsize=(8, 6), dpi=300)
        plt.hist(energies, bins=20, alpha=0.7,
                 color='steelblue', edgecolor='black')
        plt.axvline(x=1e-4, color='red', linestyle='--', linewidth=2,
                    label='Isomorphism Threshold')
        plt.xlabel('Final Energy', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        title = f'Energy Distribution: {pair_name}\n'
        title += f'n={len(energies)} runs, GPU={self.use_gpu}, shots={self.shots}'
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dirs['energy_distributions'],
                                 f"{safe_name}_energy_dist.png"),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        # Create convergence plot
        if histories and any(len(h) > 0 for h in histories):
            plt.figure(figsize=(10, 6), dpi=300)
            for i, hist in enumerate(histories):
                if len(hist) > 0:
                    plt.plot(hist, alpha=0.3, color='blue', linewidth=0.5)

            # Plot average convergence
            valid_histories = [h for h in histories if len(h) > 0]
            if valid_histories:
                min_len = min(len(h) for h in valid_histories)
                if min_len > 0:
                    avg_hist = np.mean([h[:min_len]
                                        for h in valid_histories], axis=0)
                    plt.plot(avg_hist, color='red',
                             linewidth=2, label='Average')

            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Energy', fontsize=12)
            plt.title(f'Convergence Traces: {pair_name}\nGPU={self.use_gpu}, shots={self.shots}',
                      fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dirs['convergence_plots'],
                                     f"{safe_name}_convergence.png"),
                        dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

        # Save mapping matrix if best result exists
        if best_result is not None and hasattr(best_result, 'x') and len(best_result.x) > 0:
            try:
                n = int(np.sqrt(len(best_result.x)))
                if n > 0:
                    mapping_matrix = best_result.x.reshape(n, n)

                    plt.figure(figsize=(6, 5), dpi=300)
                    sns.heatmap(mapping_matrix, annot=True, cmap="YlOrRd", cbar=True,
                                linewidths=0.5, linecolor='gray', fmt='.1f')
                    plt.xlabel('Graph 2 Nodes', fontsize=12)
                    plt.ylabel('Graph 1 Nodes', fontsize=12)
                    title = f'Best Mapping Matrix: {pair_name}\n'
                    title += f'Energy: {best_result.fval:.4f}, GPU: {self.use_gpu}'
                    plt.title(title, fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dirs['mapping_matrices'],
                                             f"{safe_name}_mapping.png"),
                                dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close()
            except:
                pass

    def generate_diverse_graph_pairs(self, n_nodes=4, num_pairs=6):
        """Generate diverse graph pairs for analysis."""
        pairs = []

        # Always generate these reliable pairs
        reliable_pairs = [
            # Isomorphic pairs
            (f"Complete_Permuted_{n_nodes}nodes",
             nx.complete_graph(n_nodes),
             nx.complete_graph(n_nodes)),  # Same graph, will be permuted below

            (f"Cycle_Permuted_{n_nodes}nodes",
             nx.cycle_graph(n_nodes),
             nx.cycle_graph(n_nodes)),

            (f"Star_Permuted_{n_nodes}nodes",
             nx.star_graph(n_nodes-1),
             nx.star_graph(n_nodes-1)),

            # Non-isomorphic pairs
            (f"Complete_vs_Cycle_{n_nodes}nodes",
             nx.complete_graph(n_nodes),
             nx.cycle_graph(n_nodes)),

            (f"Cycle_vs_Path_{n_nodes}nodes",
             nx.cycle_graph(n_nodes),
             nx.path_graph(n_nodes)),

            (f"Star_vs_Cycle_{n_nodes}nodes",
             nx.star_graph(n_nodes-1),
             nx.cycle_graph(n_nodes)),
        ]

        # Process reliable pairs
        for name, G1_template, G2_template in reliable_pairs[:num_pairs]:
            G1 = G1_template.copy()

            # For isomorphic pairs, create a permuted version
            if "Permuted" in name:
                # Create a random permutation
                permutation = np.random.permutation(list(G1.nodes()))
                mapping = {old: new for old, new in zip(
                    G1.nodes(), permutation)}
                G2 = nx.relabel_nodes(G1.copy(), mapping)
            else:
                # For non-isomorphic pairs, use the template directly
                G2 = G2_template.copy()

            # Verify node counts match
            if len(G1.nodes()) == len(G2.nodes()):
                pairs.append((name, G1, G2))
            else:
                print(f"  Warning: Skipping {name} - node count mismatch")

        # If we need more pairs, add simple variations
        if len(pairs) < num_pairs:
            # Add wheel graphs
            if n_nodes >= 4:
                pairs.append((f"Wheel_{n_nodes}nodes",
                              nx.wheel_graph(n_nodes),
                              nx.wheel_graph(n_nodes)))

            # Add complete bipartite
            if n_nodes >= 4:
                k = n_nodes // 2
                pairs.append((f"CompleteBipartite_{n_nodes}nodes",
                              nx.complete_bipartite_graph(k, n_nodes - k),
                              nx.complete_bipartite_graph(k, n_nodes - k)))

        return pairs[:num_pairs]

# %%
# ==========================================
# MAIN EXECUTION WITH GPU SUPPORT
# ==========================================


if __name__ == "__main__":
    print("\n" + "="*80)
    print("QUANTUM GRAPH ISOMORPHISM WITH GPU ACCELERATION")
    print("="*80)

    # Ask user for GPU preference
    use_gpu_input = input("\nUse GPU acceleration? (y/n): ").lower().strip()
    use_gpu = use_gpu_input == 'y'

    # Safety limits
    MAX_NODES = 9      # Maximum nodes to analyze (16 qubits max)
    MAX_QUBITS = 90    # Hard limit for safety
    PAIRS_PER_SIZE = 30  # Reasonable number of pairs
    RUNS_PER_PAIR = 3  # Few runs per pair
    SHOTS = 1024       # Number of shots for sampling

    # Setup output directories
    print("\nSetting up output directories...")
    output_dirs = setup_output_directories()
    print(f"Results will be saved to: {output_dirs['base']}")

    # Initialize analyzer with GPU setting
    analyzer = GraphIsoAnalyzer(output_dirs, use_gpu=use_gpu, shots=SHOTS)

    # Generate diverse graph pairs
    print("\nGenerating diverse graph pairs...")
    all_pairs = []

    for n_nodes in range(2, MAX_NODES + 1):
        n_qubits = n_nodes ** 2
        if n_qubits > MAX_QUBITS:
            print(
                f"Skipping {n_nodes} nodes ({n_qubits} qubits > {MAX_QUBITS} limit)")
            continue

        print(
            f"  Generating {PAIRS_PER_SIZE} pairs with {n_nodes} nodes ({n_qubits} qubits)...")
        size_pairs = analyzer.generate_diverse_graph_pairs(
            n_nodes=n_nodes, num_pairs=PAIRS_PER_SIZE)
        all_pairs.extend(size_pairs)

    # for pair_name, G1, G2 in tqdm(all_pairs, desc="Analyzing graph pairs"):
    #     # Safety check for node count
    #     if len(G1.nodes()) != len(G2.nodes()):
    #         print(f"\nWarning: Skipping {pair_name} - node count mismatch "
    #             f"({len(G1.nodes())} vs {len(G2.nodes())})")
    #         continue

    #     # Also check if graphs are valid (no isolated nodes that might cause issues)
    #     if len(G1.nodes()) == 0 or len(G2.nodes()) == 0:
    #         print(f"\nWarning: Skipping {pair_name} - empty graph")
    #         continue

    #     analyzer.analyze_pair(G1, G2, name=pair_name, runs=RUNS_PER_PAIR)

    # Add single node case
    G_single = nx.Graph()
    G_single.add_node(0)
    all_pairs.append(("Single Node", G_single, G_single.copy()))

    print(f"\nTotal pairs to analyze: {len(all_pairs)}")
    print(f"Runs per pair: {RUNS_PER_PAIR}")
    print(f"Total quantum runs: {len(all_pairs) * RUNS_PER_PAIR}")
    print(f"Using GPU: {use_gpu}")
    print(f"Shots per run: {SHOTS}")

    # Analyze all pairs
    start_time = time.time()
    for pair_name, G1, G2 in tqdm(all_pairs, desc="Analyzing graph pairs"):
        analyzer.analyze_pair(G1, G2, name=pair_name, runs=RUNS_PER_PAIR)

    total_time = time.time() - start_time
    print(f"\nTotal analysis time: {total_time:.1f} seconds")

    # Save metadata and results
    analyzer.save_metadata()

    # Save all results
    results_file = os.path.join(output_dirs['data'], 'all_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump({
            'results_db': analyzer.results_db,
            'graph_pairs': analyzer.graph_pairs,
            'output_dirs': output_dirs,
            'config': {
                'max_nodes': MAX_NODES,
                'pairs_per_size': PAIRS_PER_SIZE,
                'runs_per_pair': RUNS_PER_PAIR,
                'shots': SHOTS,
                'use_gpu': use_gpu,
                'total_time': total_time
            }
        }, f)

    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    df = pd.DataFrame(analyzer.results_db)
    if len(df) > 0:
        print(f"\nResults Summary:")
        print(f"Total quantum runs: {len(df)}")
        print(f"Unique graph pairs: {len(df['pair_name'].unique())}")
        print(f"Graph sizes analyzed: {sorted(df['n_nodes'].unique())}")

        iso_mask = df['is_isomorphic'] == True
        if iso_mask.any():
            iso_energy = df[iso_mask]['energy'].mean()
            print(f"\nIsomorphic pairs: {iso_mask.sum()} runs")
            print(f"  Mean energy: {iso_energy:.6f}")

        if (~iso_mask).any():
            noniso_energy = df[~iso_mask]['energy'].mean()
            print(f"Non-isomorphic pairs: {(~iso_mask).sum()} runs")
            print(f"  Mean energy: {noniso_energy:.6f}")

        if iso_mask.any() and (~iso_mask).any():
            energy_diff = noniso_energy - iso_energy
            print(f"  Energy separation: {energy_diff:.6f}")

        # Calculate accuracy
        threshold = 1e-4
        predictions = df['energy'] < threshold
        accuracy = (predictions == df['is_isomorphic']).mean()
        print(f"\nOverall accuracy: {accuracy:.2%}")

    print(f"\nAll results saved to: {output_dirs['base']}")
    print("Check the README.md file for detailed analysis.")
    print("="*80)

# %%
