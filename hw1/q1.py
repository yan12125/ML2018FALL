from matplotlib import pyplot as plt
import numpy

from model import Linear_Regression
from pm25_data import read_data, extract_features, extract_target


def main():
    numpy.set_printoptions(threshold=numpy.nan)

    model = Linear_Regression()

    feature_hours = 1

    data = read_data()
    features = extract_features(data, feature_hours)
    target = extract_target(data, feature_hours, pm25_row=9)

    legends = []
    for lr in (0.01, 0.02, 0.05, 0.1):
        loss_data = model.train(features, target, epochs=5000, lr=lr)
        legends.append(f'lr = {lr}')
        plt.plot(loss_data)

    plt.legend(legends, fontsize=16)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Root Mean Squared Error (RMSE)', fontsize=20)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.show()


if __name__ == '__main__':
    main()
