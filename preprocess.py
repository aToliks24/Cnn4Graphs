import networkx as nx
from collections import OrderedDict
import copy


def is_canonical(nodes_lbls_prev, nodes_lbls):
    return all([nodes_lbls[k][0] == nodes_lbls_prev[k][0] for k in nodes_lbls.keys()])


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
        nodes_lbls_prev[v] =[None, None]
        nodes_lbls[v] = [str(1).zfill(n_dig), None]
    print(nodes_lbls)

    while not is_canonical(nodes_lbls_prev, nodes_lbls):
        for v in nodes_lbls.keys():
            nbrs_l = []
            for nbr in sub_G.neighbors(v):
                nbrs_l.append(nodes_lbls[nbr][0])
            nbrs_l.sort()
            nodes_lbls[v][1] = ''.join(nbrs_l)
        nodes_lbls = OrderedDict(sorted(nodes_lbls.items(), key=lambda t: (t[1][0], t[1][1])))

        nodes_lbls_prev = copy.deepcopy(nodes_lbls) # do deep copy

        rank = 0
        prev_l = None
        for v in nodes_lbls.keys():
            if nodes_lbls[v] != prev_l:
                prev_l = nodes_lbls[v].copy()
                rank += 1
            nodes_lbls[v][0] = str(rank).zfill(n_dig)
    print(nodes_lbls)
    return list(nodes_lbls.keys())



g=nx.read_graphml('Datasets/DD/DD_1.graphml')
n=g.nodes()
sorted_nodes_by_degree=sorted(g.degree(),key=lambda tup: tup[1],reverse=True)

tmp = nx.Graph()
tmp.add_edge(1,3)
tmp.add_edge(2,3)
tmp.add_edge(3,4)
tmp.add_edge(4,5)
tmp.add_edge(5,6)
tmp.add_edge(6,3)


print(canonical_subgraph(tmp, tmp.nodes()))

print(canonical_subgraph(g,n))



