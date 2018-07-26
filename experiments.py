from keras.models import Sequential,Model
from keras.layers import Conv1D,Flatten,Dense,Dropout,Input,Concatenate
from keras.callbacks import TensorBoard,EarlyStopping
import data_generator
import numpy as np
import os
import json
np.random.seed(7)
import networkx as nx
import matplotlib.pyplot as plt

def prepare_paths(dataset_dict,overwrite=False):
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

def create_1Dcnn(k,width, num_of_classes,n_channels=1):
    model = Sequential()
    model.add(Conv1D(16, kernel_size=k, strides=width, activation='relu',input_shape=(k*width, n_channels)))
    model.add(Conv1D(8, kernel_size=10, strides=1, activation='relu',padding='same'))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])  # optimizer, metrics
    return model

def create_1DdoubleCnn2(k,w1,w2 ,num_of_classes):
    I1=Input((k*w1,1),name='vertex_input')
    C1=Conv1D(16, kernel_size=k, strides=w1, activation='relu',input_shape=(k*w1, 1))(I1)
    I2 = Input((k * w2, 1),name='edge_input')
    C2=Conv1D(16, kernel_size=k, strides=w2, activation='relu',input_shape=(k*w2, 1))(I2)
    conc = Concatenate()([C1, C2])
    C3=Conv1D(8, kernel_size=10, strides=1, activation='relu',padding='same')(conc)
    F1=Flatten()(C3)
    Dense1=Dense(128,activation='relu')(F1)
    Drop1=Dropout(0.5)(Dense1)
    SM=Dense(num_of_classes, activation='softmax')(Drop1)
    ModelAll=Model([I1,I2],SM)
    ModelAll.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return ModelAll

def get_recommended_width(name, datasets_path):
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

def plot_graph(dirname, ds_name,g1_name,g2_name,suptitle, h, k, mode, n_epochs, savefig, showfig, width,test_value):
    fig = plt.figure()
    txt = '''
    Mode: \'{}\' , K={},  Width=({}), Test {} = {}
    '''.format(mode, k, ','.join([str(w) for w in width]),suptitle,'%.3f'%test_value)
    fig.text(.1, .1, txt)
    fig.suptitle('Dataset \'{}\' {}'.format(ds_name,suptitle))
    ax1 = fig.add_axes((.1, .25, .8, .65))
    ax1.set_xlabel('Epochs')
    ax1.plot(list(range(n_epochs)), h.history[g1_name], linestyle=':', label='Validation')
    ax1.plot(list(range(n_epochs)), h.history[g2_name], linestyle='--', label='Train')
    ax1.plot(list(range(n_epochs)),[test_value]*n_epochs, linestyle='-.', label='Test Result')
    plt.legend(loc='best')
    if savefig:
        plt.savefig(dirname + '/{}.pdf'.format(suptitle))
    if showfig:
        plt.show()
    plt.clf()

Datasets_dict={
          'enzymes':{'path':'Datasets/enzymes/',
                    'labels':'Datasets/enzymes/enzymes.label',
                     'data':'Datasets/enzymes/enzymes.list'
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
          'NCI1': {'path': 'Datasets/NCI1/',
                      'labels': 'Datasets/NCI1/NCI1.label',
                      'data': 'Datasets/NCI1/NCI1.list',

                      }

          }



def train_test(ds_name, k, mode, ds_path='Datasets/', width=None, n_epochs=100, test_percent=0.20,val_percent=0.10, batch_size=20,savefig=False,showfig=True):
    data, labels=prepare_paths(Datasets_dict[ds_name], overwrite=True)
    num_of_classes=len(set(labels.values()))
    rands1 = np.random.random(len(data))
    if type(width) == int or type(width) == tuple and len(width)==1:
        wv=width
        we=width
    elif type(width) == tuple:
        wv=width[0]
        we=width[1]
    else:
        rec_width=get_recommended_width(ds_name,ds_path)
        wv=rec_width['V']
        we=rec_width['E']
        print('Chosen Recommended width values are {} for verteces and {} for edges'.format(wv,we))
    if mode=='comb':
        m=create_1DdoubleCnn2(k,wv,we,num_of_classes)
        width= (wv,we)
    elif mode=='vertex' :
        m=create_1Dcnn(k,wv,num_of_classes,n_channels=1)
        width=(wv,)
    elif mode=='edge':
        m=create_1Dcnn(k,we,num_of_classes)
        width=(we,)
    elif mode=='vertex_channels':
        m=create_1Dcnn(k,wv,num_of_classes,n_channels=4)
        width=(wv,)
    else:
        raise Exception("'mode' parameter should be in ['vertex','edge','comb','vertex_channels'] ")
    X_train_ids=data[rands1 > test_percent]
    X_test_ids = data[rands1 <= test_percent]
    rands2 = np.random.random(len(X_train_ids))
    X_val_ids=X_train_ids[rands2<= val_percent]
    X_train_ids=X_train_ids[rands2 > val_percent]


    dg_train=data_generator.DataGenerator(X_train_ids, labels, Datasets_dict[ds_name]['path'], len(set(labels.values())), width=width, k=k,
                                          mode=mode, batch_size=batch_size)
    dg_test=data_generator.DataGenerator(X_test_ids, labels, Datasets_dict[ds_name]['path'], len(set(labels.values())), width=width, k=k,
                                         mode=mode)
    dg_val=data_generator.DataGenerator(X_val_ids, labels, Datasets_dict[ds_name]['path'], len(set(labels.values())), width=width, k=k,
                                         mode=mode)
    dirname='TB_Dataset-{}__Mode-{}__K-{}__Width-{}'.format(ds_name,mode,k,'_'.join([str(w) for w in width]))
    h= m.fit_generator(dg_train,epochs=n_epochs,verbose=2,callbacks=[TensorBoard(dirname),EarlyStopping(patience=10,monitor='val_acc')],validation_data=dg_val.getallitems(),workers=1)
    X_test,y_test=dg_test.getallitems()
    ev=m.evaluate(X_test,y_test)
    with open(dirname+'/history.json', 'w') as file:
        file.write(json.dumps(h.history))
    plot_graph(dirname, ds_name,'val_acc','acc','Accuracy', h, k, mode, len(h.epoch), savefig, showfig, width,ev[1])
    plot_graph(dirname, ds_name,'val_loss','loss','Loss', h, k, mode, len(h.epoch), savefig, showfig, width,ev[0])





dataset_names=['mutag','DD','enzymes','NCI1']
modes=['vertex','edge','comb','vertex_channels']

dataset=dataset_names[0]    #choose dataset frome dataset-list
mode=modes[0]               #choose mode frome mode-list
width=None                  #None for default recommended values,
                            #for costume values use tuple (vertex_width,edge_width) if 'comb' mode, otherwise use integer
k=10                        #common values: 5,10



train_test(ds_name=dataset, k=k, mode=mode,width=width ,n_epochs=50, test_percent=0.2, batch_size=20,savefig=True,showfig=False)

