import numpy

from model import Logistic_Regression
from data import (
    read_features,
    read_target,
)


def cross_validation(model, features, target, k):
    feature_folds = numpy.split(features, k)
    target_folds = numpy.split(target, k)

    total_loss = 0
    total_accuracy = 0

    for i in range(k):
        print(f'k-fold cross validation, run {i + 1}/{k}')
        train_feature_folds = feature_folds[:i] + feature_folds[(i + 1):]
        train_target_folds = target_folds[:i] + target_folds[(i + 1):]
        test_feature_fold = feature_folds[i]
        test_target_fold = target_folds[i]

        model.train(numpy.concatenate(train_feature_folds),
                    numpy.concatenate(train_target_folds))

        test_feature_fold = model.feature_scaling(test_feature_fold, train=False)
        loss = model.CrossEntropyLoss(test_feature_fold, test_target_fold)
        accuracy = model.accuracy(test_feature_fold, test_target_fold)
        print(loss)
        print(accuracy)
        total_loss += loss
        total_accuracy += accuracy

    print(total_loss / k)
    print(total_accuracy / k)


def main():
    numpy.set_printoptions(threshold=numpy.nan)
    numpy.seterr(all='raise')

    model = Logistic_Regression()

    features = read_features()
    target = read_target()

    cross_validation(model, features, target, k=10)

    model.train(features, target, epochs=10000)

    model.save('model.npz')


if __name__ == '__main__':
    main()
