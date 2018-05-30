import networkx as nx



g=nx.read_graphml('Datasets/DD/DD_1.graphml')
n=g.nodes()
sorted_nodes_by_degree=sorted(g.degree(),key=lambda tup: tup[1],reverse=True)





