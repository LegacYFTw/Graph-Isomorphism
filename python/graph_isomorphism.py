import itertools
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from dimod import BinaryQuadraticModel, AdjVectorBQM
from qiskit.optimization import QuadraticProgram
from qiskit.optimization.converters import LinearEqualityToPenalty, InequalityToEquality, IntegerToBinary


def add_variables(qp, G1, G2):
    for u in G1.nodes:
        for i in G2.nodes:
            qp.binary_var(f"x_{u}_{i}")


def one_node_connectG1(qp, G1, G2):
    for u in G2.nodes:
        for i in G1.nodes:
            L = {}
            L[f"x_{u}_{i}"] = 1
        qp.linear_constraint(linear = L, sense='E', rhs=1)


def one_node_connectG2(qp, G1, G2):
    for i in G1.nodes:
        for u in G2.nodes:
            L = {}
            L[f"x_{u}_{i}"] = 1
        qp.linear_constraint(linear = L, sense='E', rhs=1)


def get_objective(qp):
    qdict = {}
    for key, item in qp.variables_index.items():
        qdict[item] = key

    qp.objective.quadratic.to_dict()
    Q_int = qp.objective.quadratic.to_dict()
    L_int = qp.objective.linear.to_dict()
    
    Q = defaultdict(lambda: 0) 
    for k, v in Q_int.items():
        k1 = k[0]
        k2 = k[1]
        Q[(qdict[k1], qdict[k2])] = v
    
    L = defaultdict(lambda: 0)
    for k, v in L_int.items():
        L[qdict[k]] = v

    C = qp.objective.constant
    return Q, L, C


def not_in_edges1(qp, G1, G2, C2):

    Q, L, C = get_objective(qp)
    for i,j in itertools.product(G1.nodes, G1.nodes): #original
    # for i,j in itertools.combinations(G1.nodes, 2):
        if (i,j) not in G1.edges and i != j:

            for u,v in itertools.product(G2.nodes, G2.nodes):# original
            # for u,v in itertools.combinations(G2.nodes, 2):
                if (u,v) in G2.edges:
                    Q[(f"x_{u}_{i}", f"x_{v}_{j}")] = C2
    
    qp.minimize(quadratic = Q, linear = L, constant = C)


def not_in_edges2(qp, G1, G2, C2):

    Q, L, C = get_objective(qp)
    for i,j in itertools.product(G1.nodes, G1.nodes): #original
    # for i,j in itertools.combinations(G1.nodes, 2):
        if (i,j) in G1.edges:

            for u,v in itertools.product(G2.nodes, G2.nodes): #original
            # for u,v in itertools.combinations(G2.nodes, 2):
                if (u,v) not in G2.edges and u != v:
                    Q[(f"x_{u}_{i}", f"x_{v}_{j}")] = C2
    
    qp.minimize(quadratic = Q, linear = L, constant = C)



if __name__ == "__main__":



    G1 = nx.Graph()

    # G1_nodes_list = [0,1,2,3]
    G1_edges_list = [(0,1), (2,3)]
    # G1.add_nodes_from(G1_nodes_list)
    G1.add_edges_from(G1_edges_list)
    # G1.add_edges_from([(1, 2), (1, 3)])
    # G1.add_node(4)
    nx.draw(G1, with_labels=True, alpha=1, node_size=500)
    
    print(G1.nodes)
    # nx.draw(G1, with_labels = True)
    plt.show()

    G2 = nx.Graph()
    # G2.add_edges_from([(1, 4), (2, 4)])
    # G2.add_node(3)

    G2_edges_list = [(0,3), (1,3), (2,0)]
    # G2.add_nodes_from(G2_nodes_list)
    G2.add_edges_from(G2_edges_list)
    # G2.add_node(2)
    print(G2.nodes)
    nx.draw(G2, with_labels=True, alpha=1, node_size=500)
    # nx.draw(G2, with_labels = True)
    plt.show()

    qp = QuadraticProgram()
    ineq2eq = InequalityToEquality()
    int2bin = IntegerToBinary()

    C1 = 1
    C2 = 1

    # print('Ration of: ', C2/C1)

    add_variables(qp, G1, G2)

    one_node_connectG1(qp, G1, G2)
    lineq2penalty = LinearEqualityToPenalty(penalty=C1)
    qp = lineq2penalty.convert(int2bin.convert(ineq2eq.convert(qp)))

    one_node_connectG2(qp, G1, G2)
    lineq2penalty = LinearEqualityToPenalty(penalty=C1)
    qp = lineq2penalty.convert(int2bin.convert(ineq2eq.convert(qp)))

    not_in_edges1(qp, G1, G2, C2)
    not_in_edges2(qp, G1, G2, C2)

    Q, offset = AdjVectorBQM(qp.objective.linear.to_dict(), qp.objective.quadratic.to_dict(), 
                         qp.objective.constant, vartype='BINARY').to_qubo()


    print('offset:', offset)
    import neal
    model = BinaryQuadraticModel.from_numpy_matrix(Q)

    qubo, _ = model.to_qubo()
    s = neal.SimulatedAnnealingSampler()
    sampleset = s.sample_qubo(qubo, beta_range=(5, 100), num_sweeps=20000, num_reads=100,
                            beta_schedule_type='geometric')


    energies = []
    for datum in sampleset.data(fields=["sample", "energy"]):
        energies.append(datum.energy)


    print(energies+offset)