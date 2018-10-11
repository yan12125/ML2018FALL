import csv
import sys

import numpy
from keras.models import load_model

model = load_model('model.h5')

X_test = numpy.zeros((18, 9))
with open(sys.argv[1]) as f, open(sys.argv[2], 'w') as g:
    reader = csv.reader(f)
    print('id,value', file=g)
    for idx, row in enumerate(reader):
        day, kind = divmod(idx, 18)
        data_today = row[2:]
        if row[1] == 'RAINFALL':
            data_today = list(map(lambda d: 0 if d == 'NR' else d, data_today))
        X_test[kind][:] = data_today
        if kind == 17:
            Y = model.predict(numpy.transpose(X_test))
            print('id_{},{}'.format(day, Y[-1][0]), file=g)
