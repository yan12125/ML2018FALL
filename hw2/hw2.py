import sys

from model import Logistic_Regression
from inference import inference


def main():
    model = Logistic_Regression.load('model.npz')

    inference(model, sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()
