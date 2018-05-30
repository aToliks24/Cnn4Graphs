import networkx as nx
import operator





def ReceptiveField(Graph,v,k):
    pass


def NeighAssemb(Graph,v,k):
    N=[[v]]
    ln=1
    colored_list=[]
    while ln<k and len(N) < len(Graph):
        vertexes,colored_list=get_next_bfs_layer(Graph, N[-1],colored_list)
        N.append(vertexes)
        ln += len(N[-1])
    return N




def get_next_bfs_layer(Graph, prev_layer,colored_list):
        stack_of_next_layer = []
        prev_layer=list(prev_layer)
        colored_list=list(colored_list)
        while prev_layer !=[]:
            vertex=prev_layer[0]
            prev_layer.remove(vertex)
            colored_list.append(vertex)
            stack_of_next_layer.extend( list(Graph.neighbors(vertex)))
        stack_of_next_layer=list(set(stack_of_next_layer).difference(set(colored_list)))
        return stack_of_next_layer,colored_list


def NormalizeGraph(Graph,U,labeling_func,k):
    sorted_layers=[]
    for bfs_layer in U:
        sorted_layers.extend(labeling_func(Graph,bfs_layer))
    if len(sorted_layers)<k:
        sorted_layers.extend(['d'+str(i) for i in range(k-len(sorted_layers))])
    return sorted_layers[:k]




def random_order_labeling(Graph, bfs_layer):
    return bfs_layer



def rank_comparator(Graph):
    def cmp_value(v1,v2):
        L = Graph.degree()
        if L[v1]>L[v2]:
            return True
        else:
            return False
    return cmp_value


def make_comparator(less_than):
    def compare(x, y):
        if less_than(x, y):
            return -1
        elif less_than(y, x):
            return 1
        else:
            return 0
    return compare




g=nx.read_graphml('Datasets/DD/DD_1.graphml')
n=g.nodes()
sorted_nodes_by_degree=sorted(g.degree(),key=lambda tup: tup[1],reverse=True)
na=NeighAssemb(g,'n1',500)
ng=NormalizeGraph(g, na, random_order_labeling, 500)
p=0