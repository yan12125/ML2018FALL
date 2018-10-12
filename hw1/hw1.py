import sys

import numpy

from model import Linear_Regression
from inference import inference


def main():
    model_data = numpy.load('model.npz')
    model = Linear_Regression()
    model.b = model_data['b']
    model.W = model_data['w']
    feature_hours = model_data['feature_hours']

    inference(model, feature_hours, sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()
