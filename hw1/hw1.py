import csv
import sys

import numpy

from model import Linear_Regression


def main():
    model_data = numpy.load('model.npz')
    model = Linear_Regression()
    model.b = model_data['b']
    model.W = model_data['w']
    feature_hours = model_data['feature_hours']

    data = numpy.zeros((18, feature_hours))
    with open(sys.argv[1]) as f, open(sys.argv[2], 'w') as g:
        reader = csv.reader(f)
        print('id,value', file=g)
        for idx, row in enumerate(reader):
            day, kind = divmod(idx, 18)
            data_today = row[2:]
            if row[1] == 'RAINFALL':
                data_today = list(map(lambda d: 0 if d == 'NR' else d, data_today))
            data[kind][:] = data_today[(9 - feature_hours):]
            if kind == 17:
                X_test = numpy.reshape(data.T, (18 * feature_hours, 1)).T
                Y = model.predict(X_test)
                print('id_{},{}'.format(day, Y[-1][0]), file=g)


if __name__ == '__main__':
    main()
