import csv

import numpy


def inference(model, feature_hours, test_filename, submission_filename, feature_selection=None):
    data = numpy.zeros((18, feature_hours))
    with open(test_filename) as f, open(submission_filename, 'w') as g:
        reader = csv.reader(f)
        print('id,value', file=g)
        for idx, row in enumerate(reader):
            day, kind = divmod(idx, 18)
            data_today = row[2:]
            if row[1] == 'RAINFALL':
                data_today = list(map(lambda d: 0 if d == 'NR' else d, data_today))
            data[kind][:] = data_today[(9 - feature_hours):]
            if kind == 17:
                X_test = numpy.reshape(data.T, (18 * feature_hours, 1)).T
                if feature_selection:
                    X_test = feature_selection(X_test)
                Y = model.predict(X_test)
                print('id_{},{}'.format(day, Y[-1][0]), file=g)
