import sys
n=6
seq_count = 0
# iterate all degree sequences of length 6
for seq in DegreeSequences(n) :
    list_of_graphs = graphs(n, degree_sequence = seq)
    num_chosen_graphs = 0
    # count the number of graphs selected as test cases
    for graph in list_of_graphs :
        if graph.is_connected() and graph.complement().is_connected() :
            if (not (graph.is_tree())) and (not graph.complement().is_tree()): num_chosen_graphs += 1
    if num_chosen_graphs > 1:
        list_of_graphs = graphs(n, degree_sequence = seq)
        print( str(seq), "#" , seq_count ) 
        print( str(num_chosen_graphs) )
        for graph in list_of_graphs :
            if (not (graph.is_tree())) and (not graph.complement().is_tree()): 
                if graph.is_connected() and graph.complement().is_connected():
                    print(str(graph.order()))
                    for i in graph.vertices():
                        for j in graph.neighbors(i): print(j) 
                        print
        seq_count += 1