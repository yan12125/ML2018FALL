import numpy

from model import Linear_Regression
from pm25_data import read_data, extract_features, extract_target
from inference import inference

feature_hours = 9


def run(label, features, target):
    model = Linear_Regression()

    loss_data = model.train(features, target, epochs=10000)
    print('label={}, loss={}'.format(label, loss_data[-1]))

    def feature_selection(X_test):
        if label != 'pm25_only':
            return X_test

        return X_test[..., 9::18]

    inference(model, feature_hours, './data/test.csv',
              f'./submission_q2_{label}.txt',
              feature_selection)


def main():
    numpy.set_printoptions(threshold=numpy.nan)

    data = read_data()
    features = extract_features(data, feature_hours)
    target = extract_target(data, feature_hours, pm25_row=9)

    run('pm25_only', features[..., 9::18], target)
    run('all', features, target)


if __name__ == '__main__':
    main()
