from keras.models import Sequential
from keras.layers import Conv1D,Flatten,Dense,Dropout
import preprocess
import data_generator
import time
import numpy as np

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
np.random.seed(7)



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

m=create_1Dcnn(20,5,2)
f_labels='Datasets/mutag/mutag.label'
f_data='Datasets/mutag/mutag.list'
data_dir='Datasets/mutag/'
with open(f_data,'r') as r:
    data=np.array( [d[:-1] for d in r.readlines()])

with open(f_labels,'r') as r:
    labels=[ float(l) for l in r.readlines()[0].split(' ')]
    labels={ data[i]:labels[i] for i in range(len(labels))  }
rands=np.random.random(len(data))
X_train=data[rands<=0.8]
X_test=data[rands>0.8]

dg_train=data_generator.DataGenerator(X_train,labels,data_dir)
dg_test=data_generator.DataGenerator(X_test,labels,data_dir)
data_test,labels_test=dg_test.getallitems()
m.fit_generator(dg_train,epochs=50,verbose=2)
y_pred=m.predict_classes(data_test)
acc=np.sum([i==j for i,j in zip(y_pred,labels_test)])/len(labels_test)
print(acc)