# -*- coding:utf-8 -*-
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
# fix random seed for reproducibility
numpy.random.seed(7)

# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
# print(char_to_int)
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# print(int_to_char)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 1
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    # print(seq_in)
    seq_out = alphabet[i + seq_length]
    # print(seq_out)
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
    # print(seq_in, '->', seq_out)

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (len(dataX), seq_length, 1))
# print(X)

# X = numpy.reshape(dataX, (len(dataX), 1, seq_length))
# print(X)
X = X / float(len(alphabet))
# print(X)

# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# print(y)

# create and fit the model
model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X, y, epochs=500, batch_size=1, verbose=0)
model.fit(X, y, epochs=5000, batch_size=len(dataX), verbose=2, shuffle=False)

# summarize performance of the model
scores = model.evaluate(X, y, verbose=0)
# print(scores)
print("Model Accuracy: %.2f%%" % (scores[1]*100))

# demonstrate some model predictions
for pattern in dataX:
    # print(pattern)
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    # x = numpy.reshape(pattern, (1, 1, len(pattern)))
    x = x / float(len(alphabet))
    # print(x)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(seq_in, "->", result)

## suiji 
print("Test a Random Pattern:")
for i in range(0,20):
    pattern_index = numpy.random.randint(len(dataX))
    pattern = dataX[pattern_index]
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(seq_in, "->", result)