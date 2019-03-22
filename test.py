from keras import backend as K
K.set_image_dim_ordering('th')
import numpy as np
np.random.seed(123)
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.datasets import mnist

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(X_train.shape)
import pylab
from matplotlib import pyplot as plt

# plt.figure()
# plt.imshow(X_train[0])
# plt.axis('off') # 不显示坐标轴  
# pylab.show()

# print(X_train[0])
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
# print(X_train.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# print(y_train.shape)
# print(y_train[:10])
# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
# print(Y_train.shape)
	
model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
# print(model.output_shape)
# (None, 32, 26, 26)

model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=10, verbose=1)
# Using TensorFlow backend.
# test.py:40: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(32, (3, 3), activation="relu", input_shape=(1, 28, 28...)`
#   model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
# WARNING:tensorflow:From D:\Anaconda3\envs\keras\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
# Instructions for updating:
# Colocations handled automatically by placer.
# 2019-03-22 20:19:25.301137: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
# test.py:44: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(32, (3, 3), activation="relu")`
#   model.add(Convolution2D(32, 3, 3, activation='relu'))
# WARNING:tensorflow:From D:\Anaconda3\envs\keras\lib\site-packages\keras\backend\tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
# Instructions for updating:
# Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
# test.py:58: UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.
#   batch_size=32, nb_epoch=10, verbose=1)
# WARNING:tensorflow:From D:\Anaconda3\envs\keras\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
# Instructions for updating:
# Use tf.cast instead.
# Epoch 1/10
# 60000/60000 [==============================] - 154s 3ms/step - loss: 0.2059 - acc: 0.9359
# Epoch 2/10
# 60000/60000 [==============================] - 163s 3ms/step - loss: 0.0861 - acc: 0.9744
# Epoch 3/10
# 60000/60000 [==============================] - 167s 3ms/step - loss: 0.0663 - acc: 0.9803
# Epoch 4/10
# 60000/60000 [==============================] - 168s 3ms/step - loss: 0.0559 - acc: 0.9826
# Epoch 5/10
# 60000/60000 [==============================] - 171s 3ms/step - loss: 0.0459 - acc: 0.9853
# Epoch 6/10
# 60000/60000 [==============================] - 160s 3ms/step - loss: 0.0413 - acc: 0.9879
# Epoch 7/10
# 60000/60000 [==============================] - 160s 3ms/step - loss: 0.0377 - acc: 0.9882
# Epoch 8/10
# 60000/60000 [==============================] - 163s 3ms/step - loss: 0.0349 - acc: 0.9886
# Epoch 9/10
# 60000/60000 [==============================] - 5373s 90ms/step - loss: 0.0309 - acc: 0.9905
# Epoch 10/10
# 60000/60000 [==============================] - 167s 3ms/step - loss: 0.0295 - acc: 0.9904