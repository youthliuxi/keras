# -*- coding:utf-8 -*-
from keras import backend as K
K.set_image_dim_ordering('th')
import numpy as np
np.random.seed(123)
from keras.layers import *
from keras.models import Sequential
from keras.utils import np_utils
from keras.datasets import mnist
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
path = "./mnist.npz"
f = np.load(path)
X_train, y_train = f['x_train'],f['y_train']
X_test, y_test = f['x_test'],f['y_test']
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
import pylab
from matplotlib import pyplot as plt
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',init = 'uniform', input_shape=(1,28,28)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(1 -0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(1 - 0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adamax',metrics=['accuracy'])
hist = model.fit(X_train, Y_train, batch_size=1024, epochs=100, verbose=1, validation_data=(X_test, Y_test))
log_file_name = "try_third/txt/uniform-Adamax-1024.txt"
with open(log_file_name,'w') as f:
	f.write(str(hist.history))
# score = model.evaluate(X_test, Y_test, verbose=0, batch_size=1024)
# print(score[0])
# print(score[1])
