import numpy as np
import keras
import os
import preprocess
import networkx as nx
import time
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels,data_dir, n_classes,batch_size=32, width=20,stride=1,k=5, shuffle=True):
        'Initialization'
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
        X_list = []
        y = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            curr_path=self.data_dir + ID
            if os.path.exists(curr_path+ '.npz'):
                X_list.append(np.array(np.load(curr_path+'.npz')['arr_0']))
            else:
                g=nx.read_graphml(curr_path)
                pp= preprocess.SelNodeSeq(g,preprocess.canonical_subgraph,stride=self.stride,width=self.width,k=self.k)
                np.savez_compressed(curr_path,pp)
                X_list.append(np.array(pp))

            # Store class
            y.append( self.labels[ID])
        y=np.array(y)
        X=np.vstack(X_list)

        return np.expand_dims( X,axis=2), keras.utils.to_categorical(y, num_classes=self.n_classes)