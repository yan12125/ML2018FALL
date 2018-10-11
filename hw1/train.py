import numpy

from model import Linear_Regression
from pm25_data import read_data, extract_features, extract_target


def cross_validation(model, features, target, k):
    feature_folds = numpy.split(features, k)
    target_folds = numpy.split(target, k)

    total_loss = 0

    for i in range(k):
        print(f'k-fold cross validation, run {i + 1}/{k}')
        train_feature_folds = feature_folds[:i] + feature_folds[(i + 1):]
        train_target_folds = target_folds[:i] + target_folds[(i + 1):]
        test_feature_fold = feature_folds[i]
        test_target_fold = target_folds[i]

        model.train(numpy.concatenate(train_feature_folds),
                    numpy.concatenate(train_target_folds))

        total_loss += model.RMSELoss(model.feature_scaling(test_feature_fold), test_target_fold)

    print(total_loss / k)


def main():
    numpy.set_printoptions(threshold=numpy.nan)

    model = Linear_Regression()

    feature_hours = 1

    data = read_data()
    features = extract_features(data, feature_hours)
    target = extract_target(data, feature_hours, pm25_row=9)

    # cross_validation(model, features, target, k=12)

    model.train(features, target)

    numpy.savez('model.npz', b=model.b, w=model.W, feature_hours=feature_hours)


if __name__ == '__main__':
    main()
