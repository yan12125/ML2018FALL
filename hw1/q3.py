import numpy

from model import Linear_Regression
from pm25_data import read_data, extract_features, extract_target
from inference import inference


def main():
    numpy.set_printoptions(threshold=numpy.nan)

    feature_hours = 1

    data = read_data()
    features = extract_features(data, feature_hours)
    target = extract_target(data, feature_hours, pm25_row=9)

    for regularization_term in (0, 0.01, 0.1, 10, 100, 1000):
        model = Linear_Regression(regularization_term)

        loss_data = model.train(features, target, epochs=10000)
        print('lambda={}, loss={}'.format(regularization_term, loss_data[-1]))
        inference(model, feature_hours, './data/test.csv',
                  f'./submission_q3_{regularization_term}.txt')


if __name__ == '__main__':
    main()
