import csv
import sys

import numpy

from model_generative import GenerativeModel


def main():
    model = GenerativeModel.load('model_generative.npz')

    test_filename = sys.argv[3]
    submission_filename = sys.argv[4]
    with open(test_filename) as f, open(submission_filename, 'w') as g:
        reader = csv.reader(f)
        print('id,value', file=g)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            X_test = numpy.zeros((1, model.dim))
            X_test = model.feature_scaling(X_test, train=False)
            Y = numpy.where(model.inference(X_test) > 0.5, 1, 0)
            print('id_{},{}'.format(idx - 1, Y[0][0]), file=g)


if __name__ == '__main__':
    main()
