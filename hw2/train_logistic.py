import numpy

from model_logistic import Logistic_Regression
from data import (
    read_features,
    read_target,
)


def main():
    numpy.set_printoptions(threshold=numpy.nan)
    # numpy.seterr(all='raise')

    model = Logistic_Regression()

    features = read_features(do_one_hot_encoding=True)
    target = read_target()

    model.train(features, target, epochs=10000)
    model.save('model_logistic.npz')


if __name__ == '__main__':
    main()
