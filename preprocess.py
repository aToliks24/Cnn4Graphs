import networkx as nx
import operator





def ReceptiveField(Graph,v,k):
    pass


def NeighAssemb(Graph,v,k):
    N=[[v]]
    ln=1
    colored_list=[]
    vertexes=[v]
    while ln<k:
        vertexes,colored_list=get_next_bfs_layer(Graph, vertexes,colored_list)
        ln+=len(vertexes)
        N.append(vertexes)
    return N








def get_next_bfs_layer(Graph, prev_layer,colored_list):
        stack_of_next_layer = []
        colored_list=list(colored_list)
        while prev_layer !=[]:
            vertex=prev_layer[0]
            prev_layer.remove(vertex)
            colored_list.append(vertex)
            stack_of_next_layer.extend( list(Graph.neighbors(vertex)))
        stack_of_next_layer=list(set(stack_of_next_layer).difference(set(colored_list)))
        return stack_of_next_layer,colored_list


def NormalizeGraph(Graph,U,v,labeling_func,k):
    sorted_layers=[]
    for bfs_layer in U:
        sorted_layers.extend(labeling_func(Graph,bfs_layer))
    return sorted_layers[:k]








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
na=NeighAssemb(g,'n1',10)
p=0