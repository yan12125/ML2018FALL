import sys

from dataprocessing import load_test_data

import numpy
from keras.models import load_model

data = load_test_data(sys.argv[1])

model = load_model('Model.213-0.6445.hdf5')

with open(sys.argv[2]) as f:
    print('id,label')
    for idx, test_features in enumerate(data):
        print('{},{}'.format(idx, numpy.argmax(model.predict(test_features), axis=1)[0]), file=f)
