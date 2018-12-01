import numpy
import csv


def load_test_data(filename):
    f = open(filename)
    csv_f = csv.reader(f)
    test_set_x = []
    for row in csv_f:
        if row[0] != "id":
            temp_list = []
            for pixel in row[1].split():
                temp_list.append(int(pixel))
            data = temp_list
            test_set_x.append(numpy.reshape(data, (1, 48, 48, 1)))
    return test_set_x


def load_data():
    train_x = []
    train_y = []
    val_x = []
    val_y = []

    with open("badtrainingdata.txt", "r") as text:
        ToBeRemovedTrainingData = []
        for line in text:
            ToBeRemovedTrainingData.append(int(line))
    number = 0

    f = open('data/train.csv')
    csv_f = csv.reader(f)

    for row in csv_f:
        number += 1
        if number in ToBeRemovedTrainingData or number == 1:
            continue

        if number < 28000:
            temp_list = []

            for pixel in row[1].split():
                temp_list.append(int(pixel))

            train_y.append(int(row[0]))
            train_x.append(temp_list)

        else:
            temp_list = []

            for pixel in row[1].split():
                temp_list.append(int(pixel))

            val_y.append(int(row[0]))
            val_x.append(temp_list)

    return train_x, train_y, val_x, val_y
