import csv

import numpy


one_hot_encoding_table = (
    ('1', '2'),                             # SEX
    ('0', '1', '2', '3', '4', '5', '6'),    # EDUCATION
    ('1', '2', '3'),                        # MARRIAGE
)


def read_target():
    ret = numpy.zeros((20000, 1))
    with open('data/train_y.csv') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            ret[idx - 1, ...] = row
    return ret


def one_hot_encoding_length():
    dim = 23
    for values in one_hot_encoding_table:
        dim += len(values) - 1
    return dim


def one_hot_encoding(row):
    ret = numpy.zeros((1, one_hot_encoding_length()))
    start_index_ = start_index = 1
    ret[0][0:start_index_] = row[0:start_index_]
    for col, values in enumerate(one_hot_encoding_table):
        for idx, val in enumerate(values):
            ret[0][start_index + idx] = 1 if row[start_index_ + col] == val else 0
        start_index += len(values)
    ret[0][start_index:] = row[(start_index_ + len(one_hot_encoding_table)):]
    return ret


def read_features(do_one_hot_encoding=True):
    dim = 23
    if do_one_hot_encoding:
        dim = one_hot_encoding_length()

    ret = numpy.zeros((20000, dim))
    with open('data/train_x.csv') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            if do_one_hot_encoding:
                ret[idx - 1][...] = one_hot_encoding(row)
            else:
                ret[idx - 1][...] = row
    return ret


def feature_selection(X):
    pass
