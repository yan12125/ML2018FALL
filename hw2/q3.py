import numpy

from model import Logistic_Regression
from data import (
    read_features,
    read_target,
    feature_selection,
)


def main():
    numpy.set_printoptions(threshold=numpy.nan)
    numpy.seterr(all='raise')

    features = read_features(do_one_hot_encoding=False)
    target = read_target()

    for idx in range(features.shape[1]):
        model = Logistic_Regression()

        features = feature_selection(features.copy(), idx)
        model.train(features, target, epochs=10000)
        scaled_features = model.feature_scaling(features, train=False)
        accuracy = model.accuracy(scaled_features, target)
        print(accuracy)


if __name__ == '__main__':
    main()
