import csv
import io

import numpy


hours_per_month = 20 * 24


def read_data():
    data = numpy.zeros((18, 5760))
    with io.open('data/train.csv', encoding='big5') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            day, kind = divmod(idx - 1, 18)
            data_today = row[3:]
            if row[2] == 'RAINFALL':
                data_today = list(map(lambda d: 0 if d == 'NR' else d, data_today))
            data[kind][(day * 24):((day + 1) * 24)] = data_today
    return data


def extract_features(data, feature_hours):
    features_per_month = 20 * 24 - feature_hours
    features = numpy.zeros((features_per_month * 12, 18 * feature_hours))
    for month in range(12):
        for hour in range(20 * 24 - feature_hours):
            for i in range(feature_hours):
                features[month * features_per_month + hour][(18 * i):(18 * (i + 1))] = data[..., month * hours_per_month + hour + i]
    return features


def extract_target(data, feature_hours, pm25_row):
    features_per_month = 20 * 24 - feature_hours
    target = numpy.zeros((features_per_month * 12, 1))
    for month in range(12):
        for hour in range(20 * 24 - feature_hours):
            target[month * features_per_month + hour] = data[pm25_row][month * hours_per_month + hour + feature_hours]
    return target
