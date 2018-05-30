import networkx as nx
import numpy as np



def SelNodeSeq(Graph,labeling_func,stride,width,k):
    v_sorted = sorted(g.degree(), key=lambda tup: tup[1], reverse=True)[:width]
    i=0
    j=0
    input_channel=[]
    while j< width:
        if i < len(v_sorted):
            f=ReceptiveField(Graph,labeling_func,v_sorted[i][0],k)
        else:
            f=ZeroReceptiveField(k)
        input_channel.append(f)
        i+=stride
        j+=1
    return np.array(input_channel)




def ReceptiveField(Graph,labeling_func,v,k):
    N=NeighAssemb(Graph,v,k)
    G=NormalizeGraph(Graph,N,labeling_func,k)
    return G


def ZeroReceptiveField(k):
    return [0]*k


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
    l = list(range(len(bfs_layer)))
    np.random.shuffle(l)
    return l




g=nx.read_graphml('Datasets/DD/DD_1.graphml')
n=g.nodes()
sorted_nodes_by_degree=sorted(g.degree(),key=lambda tup: tup[1],reverse=True)

#na=NeighAssemb(g,'n1',500)
#ng=NormalizeGraph(g, na, random_order_labeling, 500)
#f=ReceptiveField(g,random_order_labeling,'n1',100)
ns=SelNodeSeq(g,random_order_labeling,stride=3,width=10,k=100)
p=0