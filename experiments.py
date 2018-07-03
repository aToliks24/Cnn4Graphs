from keras.models import Sequential,Model
from keras.layers import Conv1D,Flatten,Dense,Dropout,LSTM,Input,Concatenate
from keras.callbacks import TensorBoard
import preprocess
import data_generator
import time
import matplotlib.pyplot as plt
import numpy as np
import os
np.random.seed(7)
import networkx as nx

'''
def create_2Dcnn(k,width, num_of_classes,stride):  # doesn't work
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(k, k), padding='same', activation='relu',input_shape=(k,width, 1)))
    model.add(Conv2D(8, kernel_size=(16, 6), padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
'''



def prepare_paths(dataset_dict,overwrite=True):
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

def create_1Dcnn(k,width, num_of_classes):
    model = Sequential()
    model.add(Conv1D(16, kernel_size=k, strides=width, activation='relu',input_shape=(k*width, 1)))
    model.add(Conv1D(8, kernel_size=10, strides=1, activation='relu',padding='same'))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])  # optimizer, metrics
    return model

def create_1DdoubleCnn(k,w1,w2, num_of_classes):
    I1=Input((k*w1,1),name='vertex_input')
    C1=Conv1D(16, kernel_size=k, strides=w1, activation='relu',input_shape=(k*w1, 1))(I1)
    M1=Model(I1,C1)
    I2 = Input((k * w2, 1),name='edge_input')
    C2=Conv1D(16, kernel_size=k, strides=w2, activation='relu',input_shape=(k*w2, 1))(I2)
    M2=Model(I2,C2)
    conc = Concatenate()([M1.output, M2.output])
    C3=Conv1D(8, kernel_size=10, strides=1, activation='relu',padding='same')(conc)
    F1=Flatten()(C3)
    Dense1=Dense(128,activation='relu')(F1)
    Drop1=Dropout(0.5)(Dense1)
    SM=Dense(num_of_classes, activation='softmax')(Drop1)
    ModelAll=Model([I1,I2],SM)
    ModelAll.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return ModelAll

Datasets={'enzymes':{'path':'Datasets/enzymes/',
                    'labels':'Datasets/enzymes/enzymes.label',
                     'data':'Datasets/enzymes/enzymes.list'
                     },
          'mutag': {'path': 'Datasets/mutag/',
                      'labels': 'Datasets/mutag/mutag.label',
                      'data': 'Datasets/mutag/mutag.list'
                      },
          'DD': {'path': 'Datasets/DD/',
                      'labels': 'Datasets/DD/DD.label',
                      'data': 'Datasets/DD/DD.list'
                      }
,
          'NCI1': {'path': 'Datasets/NCI1/',
                      'labels': 'Datasets/NCI1/NCI1.label',
                      'data': 'Datasets/NCI1/NCI1.list'
                      }

          }

enzymes='enzymes'
mutag='mutag'
DD='DD'
NCI1='NCI1'




curr_ds_name=mutag
k=10
width=30
n_epochs=5
test_percent=0.1
type='edge' #  'vertex' or 'edge' or 'comb'
data, labels=prepare_paths(Datasets[curr_ds_name],overwrite=True)


num_of_classes=len(set(labels.values()))
rands = np.random.random(len(data))
m=create_1Dcnn(k,width,num_of_classes)
#m=create_1DdoubleCnn(k,width,width,num_of_classes)
X_train=data[rands<=(1-test_percent)]
X_test=data[rands>test_percent]


dg_train=data_generator.DataGenerator(X_train,labels,Datasets[curr_ds_name]['path'],len(set(labels.values())),width=width,k=k,type=type)
dg_test=data_generator.DataGenerator(X_test,labels,Datasets[curr_ds_name]['path'],len(set(labels.values())),width=width,k=k,type=type)
data_test,labels_test=dg_test.getallitems()
m.fit_generator(dg_train,epochs=n_epochs,verbose=2,validation_data=dg_test.getallitems(),callbacks=[TensorBoard('TensorBoardDir/')])

#y_pred=m.predict_classes(data_test)
#y_true=[ np.where(r==1)[0][0] for r in labels_test ]
#acc=np.sum([i==j for i,j in zip(y_pred,y_true)])/len(labels_test)
#print('The accuracy is : %.4f'%acc)