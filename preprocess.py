from collections import OrderedDict
import copy


def SelNodeSeq(Graph, canonization_func, stride, width, k):
    """
    Description:
    Node Sequence Selection algorithm, create features from graph.
    input:
    Graph – the raw graph.
    canonization_func – function applying partial order on graph.
    stride - Relative distance between the selected sequences.
    width - size of the receptive field.
    k - number fof receptive fields.
    Output:
    input_channel - a feature vector for machine learning application.
    """
    v_sorted = sorted(Graph.degree(), key=lambda tup: tup[1], reverse=True)[:width]
    i = 0
    j = 0
    input_channel_nodes = []
    input_channel = []
    while j < width:
        if i < len(v_sorted):
            f = ReceptiveField(Graph, canonization_func, v_sorted[i][0], k)
        else:
            f = ZeroReceptiveField(k)
        input_channel_nodes.extend(f)
        i += stride
        j += 1
    for icn in input_channel_nodes:
        if icn in Graph:
            input_channel.append(list(Graph.nodes.data()[icn].values())[0])
        else:
            input_channel.append(0)
    return input_channel


def ReceptiveField(Graph, canonization_func, v, k):
    """
    Description:
    Creation a Receptive field vector starts with root node v in Graph.
    input:
    Graph – the raw graph.
    canonization_func – function applying partial order on graph.
    v - the root node of the receptive field.
    k - number of receptive fields
    Output:
    G - vector of one canonized receptive field
    """
    N = NeighAssemb(Graph, v, k)
    G = NormalizeGraph(Graph, N, canonization_func, k)
    return G


def ZeroReceptiveField(k):
    """
    Description:
    Padding to vector
    input:
    k - size of the vector
    Output:
    Vector of zeros
    """
    return [0] * k


def NeighAssemb(Graph, v, k):
    """
    Description:
    Building the neighborhood of node v.
    input:
    Graph – the raw graph.
    v - the root node of the receptive field.
    k - number of receptive fields
    Output:
    N - vertexes list of the neighborhood.
    """
    N = [[v]]
    ln = 1
    colored_list = []
    while ln < k and len(N) < len(Graph):
        vertexes, colored_list = get_next_bfs_layer(Graph, N[-1], colored_list)
        N.append(vertexes)
        ln += len(N[-1])
    return N


def get_next_bfs_layer(Graph, prev_layer, colored_list):
    """
    Description:
    Gets the next layer of the neigborhood.
    input:
    Graph – the raw graph.
    prev_layer - the last expanded layer of vertex neighbors.
    colored_list - the list of already members vertexes.
    Output:
    stack_of_next_layer - the next layer of bfs in the neighborbood.
    colored_list - new list of members vertexes.
    """
    stack_of_next_layer = []
    prev_layer = list(prev_layer)
    colored_list = list(colored_list)
    while prev_layer != []:
        vertex = prev_layer[0]
        prev_layer.remove(vertex)
        colored_list.append(vertex)
        stack_of_next_layer.extend(list(Graph.neighbors(vertex)))
    stack_of_next_layer = list(set(stack_of_next_layer).difference(set(colored_list)))
    return stack_of_next_layer, colored_list


def NormalizeGraph(Graph, U, canonization_func, k):
    """
    Description:
    orders a receptive field in U neighborhood layers.
    input:
    Graph – the raw graph.
    U - list of the assembeled neighborhood by layer.
    canonization_func – function applying partial order on graph.
    k - number of receptive fields
    Output:
    sorted vector of receptive field.
    """
    sorted_layers = []
    all_nodes = []
    for bfs_layer in U:
        sorted_layers.extend(canonization_func(Graph, bfs_layer))
        all_nodes.extend(bfs_layer)
    if len(sorted_layers) < k:
        sorted_layers.extend(['d' + str(i) for i in range(k - len(sorted_layers))])
    return sorted_layers[:k]



def canonical_subgraph(G, nodes):
    """
    Algorithm 1 Weisfeiler-lehman Graph Labeling
    1: input: graph G = (V, E), initial colors c0 (v) = 1 for all v ∈ V
    2: output: final colors c (v) for all v ∈ V
    3: let c (v) = c0 (v) for all v ∈ V
    4: while c (v) has not converged do
        5: for each v ∈ V do
            6: collect a multiset {c (v′) |v′ ∈ Γ(v)} containing its neighbors’
            colors
            7: sort the multiset in ascending order
            8: concatenate the sorted multiset to c (v) to generate a signature
            string s (v) = ⟨c (v), {c (v′) |v′ ∈ Γ(v)}sort⟩
        9: end for
        10: sort all s (v) in lexicographical ascending order
        11: map all s (v) to new colors 1,2,3,... sequentially; same strings
        get the same color
    12: end while
    """
    n_dig = len(str(len(nodes)))
    sub_G = G.subgraph(nodes)
    nodes_lbls = OrderedDict()
    nodes_lbls_prev = OrderedDict()
    for v in sub_G.nodes():
        nodes_lbls_prev[v] = [None, None]
        nodes_lbls[v] = [str(1).zfill(n_dig), None]

    while not is_canonical(nodes_lbls_prev, nodes_lbls):
        for v in nodes_lbls.keys():
            nbrs_l = []
            for nbr in sub_G.neighbors(v):
                nbrs_l.append(nodes_lbls[nbr][0])
            nbrs_l.sort()
            nodes_lbls[v][1] = ''.join(nbrs_l)
        nodes_lbls = OrderedDict(sorted(nodes_lbls.items(), key=lambda t: (t[1][0], t[1][1])))

        nodes_lbls_prev = copy.deepcopy(nodes_lbls)  # do deep copy

        rank = 0
        prev_l = None
        for v in nodes_lbls.keys():
            if nodes_lbls[v] != prev_l:
                prev_l = nodes_lbls[v].copy()
                rank += 1
            nodes_lbls[v][0] = str(rank).zfill(n_dig)
    res = list(nodes_lbls.keys())
    res.reverse()
    return res

def is_canonical(nodes_lbls_prev, nodes_lbls):
    return all([nodes_lbls[k][0] == nodes_lbls_prev[k][0] for k in nodes_lbls.keys()])
