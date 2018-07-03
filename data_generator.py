import numpy as np
import keras
import os
import preprocess
import networkx as nx
from xml.dom import minidom
import time
import matplotlib.pyplot as plt
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels,data_dir, n_classes,batch_size=32, width=20,stride=1,k=5, shuffle=True,type='vertex'):
        'Initialization'
        self.type = type
        self.width=width
        self.stride=stride
        self.k=k
        self.dim = (k*width,1)
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = width
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.on_epoch_end()
        if type not in ['edge','vertex','comb']:
            raise ValueError('type should be in: [\'vertex\',\'edge\',\'comb\']')




    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def getallitems(self):
        X, y = self.__data_generation(self.list_IDs)
        return X,y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        time.sleep(0.5)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_vertex_list=[]
        X_edge_list = []
        model_input=[]
        y = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            curr_path=self.data_dir + ID
            if os.path.exists(curr_path+ '_vertex.npz' ) and  self.type == 'vertex':
                X_vertex_list.append(np.array(np.load(curr_path + '_vertex.npz')['arr_0']))
            elif os.path.exists(curr_path + '_edge.npz') and self.type == 'edge':
                X_edge_list.append(np.array(np.load(curr_path+'_edge.npz')['arr_0']))
            else:
                g=nx.read_graphml(curr_path)
                pp1=None
                pp2 = None
                if self.type=='vertex':
                    pp1 = preprocess.SelNodeSeq(g, preprocess.canonical_subgraph, stride=self.stride, width=self.width,k=self.k)
                    np.savez_compressed(curr_path+'_vertex',pp1)
                    X_vertex_list.append(np.array(pp1))
                if self.type=='edge':
                    xmldoc = minidom.parse(curr_path)
                    itemlist = xmldoc.getElementsByTagName('edge')
                    lg = nx.line_graph(g)
                    for item in itemlist:
                        value=float(item.firstChild.TEXT_NODE)
                        source=item.attributes['source'].value
                        target=item.attributes['target'].value
                        if (source, target) not in lg.nodes.data():
                            lg.add_node((source, target))
                        lg.nodes.data()[(source, target)].update({'label':value})
                    [x[1].update({'value':0}) for x in lg.nodes.data() if len(x[1]) == 0]
                    pp2 = preprocess.SelNodeSeq(lg, preprocess.canonical_subgraph, stride=self.stride, width=self.width,k=self.k)
                    np.savez_compressed(curr_path+ '_edge', pp2)
                    X_edge_list.append(np.array(pp2))
            # Store class
            y.append(self.labels[ID])
        one_hot_y=keras.utils.to_categorical(y, num_classes=self.n_classes)
        if self.type=='vertex':
            model_input = np.expand_dims(np.vstack(X_vertex_list), axis=2)
        elif self.type=='edge':
            model_input=np.expand_dims(np.vstack(X_edge_list),axis=2)

        return model_input,one_hot_y


class two_input_generator_bilder(object):
    pass