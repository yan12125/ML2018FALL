import numpy

from data import (
    read_features,
    read_target,
)
from model_generative import GenerativeModel


def main():
    numpy.set_printoptions(threshold=numpy.nan)
    numpy.seterr(all='raise')

    features = read_features(do_one_hot_encoding=False)
    target = read_target()

    model = GenerativeModel()
    model.train(features, target)
    model.save('model_generative.npz')


if __name__ == '__main__':
    main()
