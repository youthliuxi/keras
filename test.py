import numpy as np
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.datasets import mnist

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(X_train.shape)
import pylab
from matplotlib import pyplot as plt
plt.figure()
plt.imshow(X_train[0])
plt.axis('off') # 不显示坐标轴  
pylab.show()  
# print(X_train[0])