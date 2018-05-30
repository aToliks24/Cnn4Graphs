from keras.models import Sequential
from keras.layers import Conv1D,Flatten,Dense,Dropout

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



def create_1Dcnn(k,width, num_of_classes):
    model = Sequential()
    model.add(Conv1D(16, kernel_size=k, strides=width, activation='linear',input_shape=(k*width, 1)))
    model.add(Conv1D(8, kernel_size=10, strides=1, activation='linear',padding='same'))
    model.add(Flatten())
    model.add(Dense(128,activation='linear'))
    model.add(Dropout(0.5))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])  # optimizer, metrics
    return model

m=create_1Dcnn(10,5,2)
