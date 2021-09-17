import sys, random
import networkx as nx
import graph_util as util

# get the degree sequence and the number of graphs that correspond to it
seq = eval(sys.stdin.readline())
print seq
num_graphs = int(sys.stdin.readline().strip())
print num_graphs

# get all graphs correspond to the degree sequence
graph_list = []
for i in range(num_graphs): 
    n = int(sys.stdin.readline().strip())
    G = nx.empty_graph(n, create_using = nx.Graph())
    for u in range(n):
        neighbors = sys.stdin.readline().split()
        for v in neighbors: 
            G.add_edge(u, int(v))
    graph_list.append(G)

# generate all qubos per graph pair
for i in range(num_graphs):
    G_1 = graph_list[i]
    for j in range(i,num_graphs):
        G_2 = graph_list[j]

        # generate random vertex permutation
        perm = list(range(G_2.order()))
        random.shuffle(perm)

        # permute G_2
        G_2 = util.vertex_permutation(G_2, perm)

        # QUBO generated with the standard formula
        (Q, n, vars_dict) = util.generate_standard_qubo(G1, G2)
        util.print_qubo(Q, n)
        print 'perm = ', perm
        print 'vars = ', vars_dict