from keras.models import Sequential
from keras.layers import Conv1D,Flatten,Dense,Dropout
import preprocess
import data_generator
import time
import numpy as np
import os
np.random.seed(7)


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



def prepare_paths(dataset_dict):
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

    filelist = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
    for f in filelist:
        os.remove(os.path.join(data_dir, f))
    return data, labels

def create_1Dcnn(k,width, num_of_classes):
    model = Sequential()
    model.add(Conv1D(16, kernel_size=k, strides=width, activation='linear',input_shape=(k*width, 1)))
    model.add(Conv1D(8, kernel_size=10, strides=1, activation='linear',padding='same'))
    model.add(Flatten())
    model.add(Dense(128,activation='linear'))
    model.add(Dropout(0.5))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')  # optimizer, metrics
    return model


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


k=20
width=5
n_epochs=30
test_percent=0.2
data, labels=prepare_paths(Datasets[curr_ds_name])


num_of_classes=len(set(labels.values()))
rands = np.random.random(len(data))
m=create_1Dcnn(k,width,num_of_classes)
X_train=data[rands<=(1-test_percent)]
X_test=data[rands>test_percent]


dg_train=data_generator.DataGenerator(X_train,labels,Datasets[curr_ds_name]['path'],len(set(labels.values())),width=width,k=k)
dg_test=data_generator.DataGenerator(X_test,labels,Datasets[curr_ds_name]['path'],len(set(labels.values())),width=width,k=k)
data_test,labels_test=dg_test.getallitems()
m.fit_generator(dg_train,epochs=n_epochs,verbose=2)
y_pred=m.predict_classes(data_test)
y_true=[ np.where(r==1)[0][0] for r in labels_test ]
acc=np.sum([i==j for i,j in zip(y_pred,y_true)])/len(labels_test)
print('The accuracy is : %.4f'%acc)