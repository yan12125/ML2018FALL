import os.path
import sys

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adagrad

sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # flake8: noqa

from pm25_data import read_data, extract_features, extract_target

numpy.set_printoptions(threshold=numpy.nan)

feature_hours = 1

data = read_data()
features = extract_features(data, feature_hours)
target = extract_target(data, feature_hours, pm25_row=9)

model = Sequential()
linear_layer = Dense(units=1, input_dim=18,
                     kernel_initializer='zeros', bias_initializer='zeros')
model.add(linear_layer)
optimizer = Adagrad(lr=0.1, epsilon=0)
model.compile(loss='mse', optimizer=optimizer)

# Training
for step in range(10000):
    cost = model.train_on_batch(features, target.T[0])
    weights, biases = linear_layer.get_weights()
    print("After %d trainings, the cost: %f" % (step, cost))

model.save('model.h5')
