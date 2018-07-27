import numpy as np
import keras
import os
import preprocess
import networkx as nx
from xml.dom import minidom
import time
# import matplotlib.pyplot as plt
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, data_dir, n_classes, batch_size=32, W=20, stride=1, k=5, shuffle=True, mode='vertex', seed=7):
        """
        Description:
        Initializes a keras.utils.Sequence compatible generator for graph classification.
        input:
        list_IDs - list of paths of graphml dataset files.
        labels - labels corresponding to the list_IDs files.
        data_dir - the directory contains the dataset.
        n_classes - number of classes.
        batch_size - batch size.
        W - size of receptive field , the number of relative graph vertexes inputs into one kernel cnn kernel.
        stride - stride parameter for the Patchy-San algorithm.
        K - number of receptive fields inputs to the model size of W each one.
        shuffle - parameter controls whether to shuffle the batch bbedore feeding the classifier.
        mode - the type of features fed to the classifier. should be in ['vertex','edge','comb','vertex_channels'].
        seed - seed number for reproduction experiments.
        """
        self.mode = mode
        self.width=W
        self.stride=stride
        self.k=k
        channels = 1
        if mode=='vertex_channels':
            channels=4
        self.dim = (k * sum(W), channels)
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = W
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.on_epoch_end()
        np.random.seed(seed)
        if mode not in ['edge', 'vertex', 'comb', 'vertex_channels']:
            raise ValueError('mode should be in: [\'vertex\',\'edge\',\'comb\' \'vertex_channels\']')

    def __len__(self):
        """
        Description:
        Denotes the number of batches per epoch
        Output:
        Number of train examples per epoch.
        """

        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Description:
        Generates one batch of data by index
        Input:
        index - index of the batch.
        Output:
        X - the data features of the batch.
        y - the labels of the batch.
        """
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def getallitems(self):
        """
        Description:
        Gets a whole dataset data
        Output:
        X - the data features of the dataset.
        y - the labels of the dataset.
        """
        X, y = self.__data_generation(self.list_IDs)
        return X,y

    def on_epoch_end(self):
        """
        Description:
        Shuffles and updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        time.sleep(0.5)

    def __data_generation(self, list_IDs_temp):
        """
        Description:
        Generates data of the batch to feed the classifier
        Output:
        X - the data features of the dataset.
        y - the labels of the dataset.
        """
        X_vertex_list=[]
        X_edge_list = []
        X_vertex_channel_list=[]
        model_input=[]
        y = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            curr_path=self.data_dir + ID
            if self.mode == 'vertex' and os.path.exists(curr_path+ '_vertex.npz' ) :
                X_vertex_list.append(np.array(np.load(curr_path + '_vertex.npz')['arr_0']))
            elif self.mode == 'edge' and os.path.exists(curr_path + '_edge.npz')   :
                X_edge_list.append(np.array(np.load(curr_path+'_edge.npz')['arr_0']))
            elif self.mode == 'comb'  and os.path.exists(curr_path + '_edge.npz')and os.path.exists(curr_path+ '_vertex.npz' )  :
                X_vertex_list.append(np.array(np.load(curr_path + '_vertex.npz')['arr_0']))
                X_edge_list.append(np.array(np.load(curr_path+'_edge.npz')['arr_0']))
            elif os.path.exists(curr_path + '_vc.npz') and self.mode == 'vertex_channels':
                X_vertex_channel_list.append(np.array(np.load(curr_path + '_vc.npz')['arr_0']))
            else:
                if self.mode=='edge':
                    edge_width = self.width[0]
                elif self.mode in ['vertex','vertex_channels']:
                    vertex_width = self.width[0]
                else:
                    vertex_width = self.width[0]
                    edge_width = self.width[1]
                if self.mode in ['vertex','vertex_channels','comb']:
                    g = nx.read_graphml(curr_path)
                    pp1 = preprocess.SelNodeSeq(g, preprocess.canonical_subgraph, stride=self.stride, width=vertex_width,k=self.k)
                    if not self.mode == 'vertex_channels':
                        X_vertex_list.append(np.array(pp1))
                        np.savez_compressed(curr_path + '_vertex', pp1)
                    else:
                        cs = nx.closeness_centrality(g)
                        nx.set_node_attributes(g, cs, 'label')
                        pp1_1 = preprocess.SelNodeSeq(g, preprocess.canonical_subgraph, stride=self.stride, width=vertex_width,k=self.k)

                        bs = nx.betweenness_centrality(g)
                        nx.set_node_attributes(g, bs, 'label')
                        pp1_2 = preprocess.SelNodeSeq(g, preprocess.canonical_subgraph, stride=self.stride, width=vertex_width,k=self.k)

                        ds = nx.degree_centrality(g)
                        nx.set_node_attributes(g, ds, 'label')
                        pp1_3 = preprocess.SelNodeSeq(g, preprocess.canonical_subgraph, stride=self.stride, width=vertex_width,k=self.k)

                        voodoo_python= [np.array(x) for x in [ pp1,pp1_1,pp1_2,pp1_3]]
                        vc_arr=np.stack(voodoo_python,axis=1)
                        np.savez_compressed(curr_path + '_vc', vc_arr)
                        X_vertex_channel_list.append(vc_arr)
                if self.mode== 'edge' or self.mode== 'comb':
                    g = nx.read_graphml(curr_path)
                    lg = self.vertexes_to_edges_graph(curr_path, g)
                    pp2 = preprocess.SelNodeSeq(lg, preprocess.canonical_subgraph, stride=self.stride, width=edge_width,k=self.k)
                    np.savez_compressed(curr_path+ '_edge', pp2)
                    X_edge_list.append(np.array(pp2))


            # Store class
            y.append(self.labels[ID])
        one_hot_y=keras.utils.to_categorical(y, num_classes=self.n_classes)
        if self.mode== 'vertex':
            model_input = np.expand_dims(np.vstack(X_vertex_list), axis=2)
        elif self.mode== 'edge':
            model_input=np.expand_dims(np.vstack(X_edge_list),axis=2)
        elif self.mode== 'comb':
            model_input={'vertex_input':np.expand_dims(np.vstack(X_vertex_list), axis=2),'edge_input':np.expand_dims(np.vstack(X_edge_list),axis=2)}
        elif self.mode== 'vertex_channels':
            model_input=np.asanyarray(X_vertex_channel_list)
        return model_input,one_hot_y

    def vertexes_to_edges_graph(self, curr_path, v_graph):
        """
        Description:
        Converts vertex to edges in a graph.
        Input:
        curr_path - path of a current graphml file
        v_graph - the vertex graph
        Output:
        e_graph - edges graph.
        """
        xmldoc = minidom.parse(curr_path)
        itemlist = xmldoc.getElementsByTagName('edge')
        e_graph = nx.line_graph(v_graph)
        for item in itemlist:
            value = float(item.firstChild.TEXT_NODE)
            source = item.attributes['source'].value
            target = item.attributes['target'].value
            if (source, target) not in e_graph.nodes.data():
                e_graph.add_node((source, target))
            e_graph.nodes.data()[(source, target)].update({'label': value})
        [x[1].update({'value': 0}) for x in e_graph.nodes.data() if len(x[1]) == 0]
        return e_graph


