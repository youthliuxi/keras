# -*- coding:utf-8 -*-
import os
# os.system("activate keras")
# os.system("python test.py > test.txt")
# print("over")
# 
batch_sizes = [16,32,64,128]
optimizers = ["SGD","RMSprop","Adagrad","Adadelta","Adam","Adamax","Nadam"]
for optimizer in optimizers:
	for batch_size in batch_sizes:
		fp_name = "try_first/%s_%d.py" % (optimizer, batch_size)
		fp=open(fp_name, "w")
		pre_code = '''# -*- coding:utf-8 -*-
from keras import backend as K
K.set_image_dim_ordering('th')
import numpy as np
np.random.seed(123)
from keras.layers import *
from keras.models import Sequential
from keras.utils import np_utils
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
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
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1,28,28)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='%s',metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=%d, nb_epoch=10, verbose=2, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=%d)
print(score[0])
print(score[1])
''' % (optimizer,batch_size,batch_size)
		fp.write(pre_code)
		fp.close()

for optimizer in optimizers:
	for batch_size in batch_sizes:
		file_path = "try_first"
		fp_py = "%s/%s_%d.py" % (file_path,optimizer, batch_size)
		fp_txt = "%s/txt/%s_%d.txt" % (file_path,optimizer, batch_size)
		os.system("python %s > %s" % (fp_py,fp_txt))

