import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.utils import np_utils
import numpy
from keras.preprocessing.image import ImageDataGenerator
import dataprocessing


def model_generate():
    model = Sequential()
    model.add(Dense(64, input_shape=(48, 48, 1)))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(Dropout(0.2))

    model.add(Dense(7))

    model.add(Activation('softmax'))

    ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=ada,
                  metrics=['accuracy'])
    model.summary()
    return model


img_rows, img_cols = 48, 48
batch_size = 128
nb_classes = 7
nb_epoch = 400
img_channels = 1

Train_x, Train_y, Val_x, Val_y = dataprocessing.load_data()

Train_x = numpy.asarray(Train_x)
Train_x = Train_x.reshape(Train_x.shape[0], img_rows, img_cols)

Val_x = numpy.asarray(Val_x)
Val_x = Val_x.reshape(Val_x.shape[0], img_rows, img_cols)

Train_x = Train_x.reshape(Train_x.shape[0], img_rows, img_cols, 1)
Val_x = Val_x.reshape(Val_x.shape[0], img_rows, img_cols, 1)

Train_x = Train_x.astype('float32')
Val_x = Val_x.astype('float32')


Train_y = np_utils.to_categorical(Train_y, nb_classes)
Val_y = np_utils.to_categorical(Val_y, nb_classes)


model = model_generate()

filepath = 'Model-DNN.{epoch:02d}-{val_acc:.4f}.hdf5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(Train_x)

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001, verbose=1)

model.fit_generator(datagen.flow(Train_x, Train_y,
                    batch_size=batch_size),
                    samples_per_epoch=Train_x.shape[0],
                    nb_epoch=nb_epoch,
                    validation_data=(Val_x, Val_y),
                    callbacks=[reduce_lr])
