from keras.models import Sequential, Model
from keras.layers import Conv1D, Flatten, Dense, Dropout, Input, Concatenate
from keras.callbacks import TensorBoard, EarlyStopping
import data_generator
import numpy as np
import os
import json

np.random.seed(7)
import networkx as nx
import matplotlib.pyplot as plt




def prepare_paths(dataset_dict, overwrite=False):
    """
    Description:
    Retrieves the paths of the files using the IDs file.
    input:
    dataset_dict – the dictionary holds the path of dataset, the IDs file and the Labels file.
    overwrite – this parameter controls whether delete the temporary cached files by generator or not.
    Output:
    data - list of paths of the dataset files
    labels - labels coresponding to the paths file
    """
    data_dir = dataset_dict['path']
    f_data = dataset_dict['data']
    f_labels = dataset_dict['labels']
    labels_to_index = {}
    with open(f_data, 'r') as r:
        data = np.array([d[:-1] for d in r.readlines()])
    with open(f_labels, 'r') as r:
        labels = [float(l) for l in r.readlines()[0].split(' ')]
        count = 0
        for l in labels:
            if l not in labels_to_index:
                labels_to_index[l] = count
                count += 1
        labels = {data[i]: labels_to_index[labels[i]] for i in range(len(labels))}
    if overwrite:
        filelist = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
        for f in filelist:
            os.remove(os.path.join(data_dir, f))
    return data, labels


def create_1Dcnn(K, W, num_of_classes, n_channels=1):
    """
    Description:
    Creation of 1D CNN model with one input for the graph classification
    input:
    W - size of receptive field , the number of relative graph vertexes inputs into one kernel cnn kernel.
    K - number of receptive fields inputs to the model size of W each one.
    num_of_classes –the number of classes that the dataset consist of.
    n_channels - number of features representing one node.
    Output:
    model - CNN model for graph classification designed by the the paper authors.
    """
    model = Sequential()
    model.add(Conv1D(16, kernel_size=K, strides=W, activation='relu', input_shape=(K * W, n_channels)))
    model.add(Conv1D(8, kernel_size=10, strides=1, activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_val))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])  # optimizer, metrics
    return model


def create_1DdoubleCnn2(k, w1, w2, num_of_classes):
    """
    Description:
    Creation of 1D CNN model with two inputs for the graph classification.
    Our design take the vertexes and the edges graphs for classification features.
    input:
    W1 - size of receptive field , the number of relative graph verteces inputs into one kernel cnn kernel.
    W2 - size of receptive field , the number of relative graph edges inputs into one kernel cnn kernel.
    K - number of receptive fields inputs to the model size of W each one.
    num_of_classes –the number of classes that the dataset consist of.
    n_channels - number of features representing one node.
    Output:
    model - CNN model for graph classification purposed by us.
    """
    I1 = Input((k * w1, 1), name='vertex_input')
    C1 = Conv1D(16, kernel_size=k, strides=w1, activation='relu', input_shape=(k * w1, 1),name='Vertexes-Convolution')(I1)
    I2 = Input((k * w2, 1), name='edge_input')
    C2 = Conv1D(16, kernel_size=k, strides=w2, activation='relu', input_shape=(k * w2, 1),name='Edges-Convolution')(I2)
    conc = Concatenate()([C1, C2])
    C3 = Conv1D(8, kernel_size=10, strides=1, activation='relu', padding='same',name='Combining-Conv')(conc)
    F1 = Flatten()(C3)
    Dense1 = Dense(128, activation='relu',name='Dense')(F1)
    Drop1 = Dropout(dropout_val)(Dense1)
    SM = Dense(num_of_classes, activation='softmax',name= 'Softmax-Layer')(Drop1)
    model = Model([I1, I2], SM)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_recommended_width(name, datasets_path):
    """
    Description:
    Calculates the average number of edges and vertexes of graphs in the current dataset.
    These values are the recommended W parameter for the Pathchy-San algorithm.
    input:
    name – name of the dataset as the directory containing the dataset.
    datasets_path - the path of the datased.
    Output:
    dictionary with average numbber of edges and vertexes.
    """
    if datasets_path[-1] != '/':
        datasets_path += '/'
    ds_path = datasets_path + name + '/'
    ds_list_path = ds_path + name + '.list'
    with open(ds_list_path, 'r') as f:
        fnames = f.readlines()
    e_sum = 0
    v_sum = 0
    count = len(fnames)
    for path in fnames:
        g = nx.read_graphml(ds_path + path.strip())
        e_sum += len(g.edges)
        v_sum += len(g.nodes)
    e_avr = int(e_sum / count)
    v_avr = int(v_sum / count)
    return {'E': e_avr, 'V': v_avr}


def plot_graph(dirname, ds_name, g1_name, g2_name, title, h, k, mode, n_epochs, savefig, showfig, width, test_value):
    """
    Description:
    Plots a graph of a model evaluation method by epochs.
    input:
    dirname - name of the directory contatins the history.json file, contains the metrics over epochs.
    ds_name - name of the dataset for the title.
    g1_name – name of metric 1 as appears in the history.json file.
    g2_name – name of metric 2 as appears in the history.json file.
    title - name of the metric for the title.
    h - history object of the run
    k - the parameter k of the current experiment for the graph description.
    mode - the mode chosen to train the model, for the graph description.
    n_epochs - the number of epochs.
    savefig - parameter controls whether saving the graph to pdf file.
    showfig - parameter controls whether showing the graph automatically afrer run.
    width -  the parameter W of the current experiment for the graph description.
    Output:
    model - CNN model for graph classification designed by the the paper authors.
    test_value - the metric value for the test-set evaluation.
    """
    fig = plt.figure()
    txt = '''
    Mode: \'{}\' , K={},  Width=({}), Test {} = {}
    '''.format(mode, k, ','.join([str(w) for w in width]), title, '%.3f' % test_value)
    fig.text(.1, .1, txt)
    fig.suptitle('Dataset \'{}\' {}'.format(ds_name, title))
    ax1 = fig.add_axes((.1, .25, .8, .65))
    ax1.set_xlabel('Epochs')
    ax1.plot(list(range(n_epochs)), h.history[g1_name], linestyle=':', label='Validation')
    ax1.plot(list(range(n_epochs)), h.history[g2_name], linestyle='--', label='Train')
    ax1.plot(list(range(n_epochs)), [test_value] * n_epochs, linestyle='-.', label='Test Result')
    plt.legend(loc='best')
    if savefig:
        plt.savefig(dirname + '/{}.pdf'.format(title))
    if showfig:
        plt.show()
    plt.clf()


Datasets_dict = {
    'enzymes': {'path': 'Datasets/enzymes/',
                'labels': 'Datasets/enzymes/enzymes.label',
                'data': 'Datasets/enzymes/enzymes.list'
                },
    'mutag': {'path': 'Datasets/mutag/',
              'labels': 'Datasets/mutag/mutag.label',
              'data': 'Datasets/mutag/mutag.list',

              },
    'DD': {'path': 'Datasets/DD/',
           'labels': 'Datasets/DD/DD.label',
           'data': 'Datasets/DD/DD.list'

           }
    ,
    'ptc':{'path': 'Datasets/ptc/',
           'labels': 'Datasets/ptc/ptc.label',
           'data': 'Datasets/ptc/ptc.list'
    },
    'proteins':{'path': 'Datasets/proteins/',
           'labels': 'Datasets/proteins/proteins.label',
           'data': 'Datasets/proteins/proteins.list'
    },

    'NCI1': {'path': 'Datasets/NCI1/',
             'labels': 'Datasets/NCI1/NCI1.label',
             'data': 'Datasets/NCI1/NCI1.list',

             },
    'collab': {'path': 'Datasets/collab/',
               'labels': 'Datasets/collab/collab.label',
               'data': 'Datasets/collab/collab.list',

               },
    'imdb_action_romance': {'path': 'Datasets/imdb_action_romance/',
                            'labels': 'Datasets/imdb_action_romance/imdb_action_romance.label',
                            'data': 'Datasets/imdb_action_romance/imdb_action_romance.list',

                            },
    'reddit_iama_askreddit_atheism_trollx': {'path': 'Datasets/reddit_iama_askreddit_atheism_trollx/',
                                             'labels': 'Datasets/reddit_iama_askreddit_atheism_trollx/reddit_iama_askreddit_atheism_trollx.label',
                                             'data': 'Datasets/reddit_iama_askreddit_atheism_trollx/reddit_iama_askreddit_atheism_trollx.list',

                                             },
    'reddit_multi_5K': {'path': 'Datasets/reddit_multi_5K/',
                        'labels': 'Datasets/reddit_multi_5K/reddit_multi_5K.label',
                        'data': 'Datasets/reddit_multi_5K/reddit_multi_5K.list',

                        }
    ,
    'imdb_comedy_romance_scifi': {'path': 'Datasets/imdb_comedy_romance_scifi/',
                                  'labels': 'Datasets/imdb_comedy_romance_scifi/imdb_comedy_romance_scifi.label',
                                  'data': 'Datasets/imdb_comedy_romance_scifi/imdb_comedy_romance_scifi.list',

                                  }

}


def train_test(ds_name, K, mode, ds_path='Datasets/', W=None, max_epochs=100, test_percent=0.20, val_percent=0.10,
               batch_size=20, savefig=False, showfig=True):
    """
    Description:
    Training and Evaluating the chosen model on a chosen dataset.
    input:
    ds_name – dataset name , as the directory of the dataset.
    K - number of receptive fields inputs to the model size of W each one.
    mode - the type of features fed to the classifier. should be in ['vertex','edge','comb','vertex_channels'].
    ds_path - the path containing the dataset directory.
    W - size of receptive field , the number of relative graph vertexes inputs into one kernel cnn kernel.
        should be NaN for recommanded values, integer for costume value of tuple for 'comb' mode.
    max_epochs - numbebr of maximum numbert of epochs.
    test_percent - the test set percent from the whole dataset.
    val_percent - the validation pervent from the train set.
    batch_size - batch size.
    savefig - parameter controls whether saving the graph to pdf file.
    showfig - parameter controls whether showing the graph automatically afrer run.
    Output:
    A trained model
    """

    data, labels = prepare_paths(Datasets_dict[ds_name], overwrite=True)
    num_of_classes = len(set(labels.values()))
    rands1 = np.random.random(len(data))
    if type(W) == int or type(W) == tuple and len(W) == 1:
        wv = W
        we = W
    elif type(W) == tuple:
        wv = W[0]
        we = W[1]
    else:
        rec_width = get_recommended_width(ds_name, ds_path)
        wv = rec_width['V']
        we = rec_width['E']
        print('Chosen Recommended width values are {} for verteces and {} for edges'.format(wv, we))
    if mode == 'comb':
        m = create_1DdoubleCnn2(K, wv, we, num_of_classes)
        W = (wv, we)
    elif mode == 'vertex':
        m = create_1Dcnn(K, wv, num_of_classes, n_channels=1)
        W = (wv,)
    elif mode == 'edge':
        m = create_1Dcnn(K, we, num_of_classes)
        W = (we,)
    elif mode == 'vertex_channels':
        m = create_1Dcnn(K, wv, num_of_classes, n_channels=4)
        W = (wv,)
    else:
        raise Exception("'mode' parameter should be in ['vertex','edge','comb','vertex_channels'] ")
    X_train_ids = data[rands1 > test_percent]
    X_test_ids = data[rands1 <= test_percent]
    rands2 = np.random.random(len(X_train_ids))
    X_val_ids = X_train_ids[rands2 <= val_percent]
    X_train_ids = X_train_ids[rands2 > val_percent]

    dg_train = data_generator.DataGenerator(X_train_ids, labels, Datasets_dict[ds_name]['path'],
                                            len(set(labels.values())), W=W, k=K,
                                            mode=mode, batch_size=batch_size)
    dg_test = data_generator.DataGenerator(X_test_ids, labels, Datasets_dict[ds_name]['path'],
                                           len(set(labels.values())), W=W, k=K,
                                           mode=mode)
    dg_val = data_generator.DataGenerator(X_val_ids, labels, Datasets_dict[ds_name]['path'], len(set(labels.values())),
                                          W=W, k=K,
                                          mode=mode)
    dirname = 'TB_Dataset-{}__Mode-{}__K-{}__Width-{}'.format(ds_name, mode, K, '_'.join([str(w) for w in W]))
    h = m.fit_generator(dg_train, epochs=max_epochs, verbose=2,
                        callbacks=[TensorBoard(dirname), EarlyStopping(patience=10, monitor='val_acc')],
                        validation_data=dg_val.getallitems(), workers=1)
    X_test, y_test = dg_test.getallitems()
    ev = m.evaluate(X_test, y_test)
    with open(dirname + '/history.json', 'w') as file:
        file.write(json.dumps(h.history))
    plot_graph(dirname, ds_name, 'val_acc', 'acc', 'Accuracy', h, K, mode, len(h.epoch), savefig, showfig, W, ev[1])
    plot_graph(dirname, ds_name, 'val_loss', 'loss', 'Loss', h, K, mode, len(h.epoch), savefig, showfig, W, ev[0])
    return m

if __name__=='__main__':
    dataset_names = ['mutag', 'DD', 'enzymes', 'NCI1', 'collab', 'imdb_action_romance',
                     "reddit_iama_askreddit_atheism_trollx", "reddit_multi_5K", "imdb_comedy_romance_scifi",'ptc']
    modes = ['vertex', 'edge', 'comb', 'vertex_channels']

    dropout_val = 0.5   #dropout
    dataset = 'mutag'   # choose dataset frome dataset-list
    mode = modes[0]     # choose mode frome mode-list
    width = None        # None for default recommended values,
                        # for costume values: if 'comb' mode use tuple (vertex_width,edge_width)
                        #                     otherwise use integer
    k = 10              # common values: 5,10
    train_test(ds_name=dataset, K=k, mode=mode, W=width, max_epochs=50, test_percent=0.2, batch_size=20, savefig=False,
               showfig=True)

