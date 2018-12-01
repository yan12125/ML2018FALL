from dataprocessing import load_data

import numpy
from keras.models import load_model

_, _, val_x, val_y = load_data()

model = load_model('Model.213-0.6445.hdf5')

prediction = []
for idx, test_features in enumerate(val_x):
    prediction.append(
        numpy.argmax(model.predict(numpy.reshape(test_features, (1, 48, 48, 1))), axis=1)[0]
    )

print(val_y)
print(prediction)
