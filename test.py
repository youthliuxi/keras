# -*- coding:utf-8 -*-
from keras import backend as K
K.set_image_dim_ordering('th')
import numpy as np
np.random.seed(123)
from keras.layers import *
# from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.datasets import mnist

# 将打乱的mnist数据集载入训练集和测试集中
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(X_train.shape)
import pylab
from matplotlib import pyplot as plt

# plt.figure()
# plt.imshow(X_train[0])
# plt.axis('off') # 不显示坐标轴  
# pylab.show()

# print(X_train[0])
path = "./mnist.npz"
f = np.load(path)
X_train, y_train = f['x_train'],f['y_train']
X_test, y_test = f['x_test'],f['y_test']
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
# 将输入的维度重构成指定为度
# (1层，28行，28列)
# print(X_train.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# 数据类型转换
X_train /= 255
X_test /= 255
# X_train = X_train / 255
# 数值归一化处理，将0~255缩放至0~1区间

# print(y_train.shape)
# print(y_train[:10])
# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
# print(Y_train.shape)
	
model = Sequential()
# 新建模型
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1,28,28)))
# 往模型里添加层
# print(model.output_shape)
# (None, 32, 26, 26)

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
# Conv2D，二维卷积层
# padding参数，补0策略，valid大小可以不同；same大小相同
model.add(MaxPooling2D(pool_size=(2,2)))
# 最大统计量池化，分为一二三，三种维度
# 还有一种是AveragePooling
model.add(Dropout(0.25))

model.add(Flatten())
# Flatten将一个维度大于或等于3的高维矩阵压扁为2维矩阵
# 保留第一个维度，将剩下的维度均作为矩阵的第二个维度
model.add(Dense(128, activation='relu'))
# Dense全连接层；activation应用激活函数relu
model.add(Dropout(0.5))
# Dropout神经元随机以一定比例失活，防止过拟合
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# 配置学习过程：loss损失函数，optimizer优化器，metrics评估标准
model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=2, verbose=2, validation_data=(X_test, Y_test))
# nb_epoch参数，所有样本的训练次数
# verbose，日志显示，0为不显示，1为显示进度条记录，2为每个epochs输出一行记录
# valiation_split，切割输入数据的一定比例作为验证集，0~1浮点数
# validation_data=(X_test, Y_test)，验证数据，计算loss
# 
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=32)
print(score[0])
print(score[1])
# 运行结果展示：
# Using TensorFlow backend.
# WARNING:tensorflow:From D:\Anaconda3\envs\keras\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
# Instructions for updating:
# Colocations handled automatically by placer.
# 2019-03-23 17:41:41.856537: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
# WARNING:tensorflow:From D:\Anaconda3\envs\keras\lib\site-packages\keras\backend\tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
# Instructions for updating:
# Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
# test.py:73: UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.
#   batch_size=32, nb_epoch=2, verbose=1)
# WARNING:tensorflow:From D:\Anaconda3\envs\keras\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
# Instructions for updating:
# Use tf.cast instead.
# Epoch 1/2
# 60000/60000 [==============================] - 120s 2ms/step - loss: 0.2064 - acc: 0.9365
# Epoch 2/2
# 60000/60000 [==============================] - 114s 2ms/step - loss: 0.0868 - acc: 0.9739
# 10000/10000 [==============================] - 5s 547us/step
# [0.04147695766454563, 0.985]
