import logging
import os
import os.path
import pickle
import sys

import jieba
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, Flatten, Embedding, LSTM, MaxPooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from util import parse_sentences

logger = logging.getLogger(__name__)

use_jieba = int(os.getenv('USE_JIEBA', 1))
use_rnn = int(os.getenv('USE_RNN', 1))


def read_data(train_x_path, train_y_path, test_x_path, dict_txt_big, use_jieba):
    PREPROCESSED_FILENAME = os.path.join('preprocessed.pickle')

    try:
        with open(PREPROCESSED_FILENAME, 'rb') as f:
            ret = pickle.load(f)
            return ret
    except FileNotFoundError:
        pass

    train_x = parse_sentences(train_x_path, use_jieba)
    test_x = parse_sentences(test_x_path, use_jieba)

    train_y = []
    with open(train_y_path) as f_y:
        for row_y in f_y:
            row_y = row_y.split(',', maxsplit=1)
            if row_y[0] == 'id':
                continue
            train_y.append(int(row_y[1]))

    ret = (train_x, train_y, test_x)
    with open(PREPROCESSED_FILENAME, 'wb') as f:
        pickle.dump(ret, f)

    return ret


def load_word2vec_model(sentences):
    WORD2VEC_MODEL_FILENAME = os.path.join('word2vec.model')

    try:
        model = Word2Vec.load(WORD2VEC_MODEL_FILENAME)
    except FileNotFoundError:
        logger.info('Train word2vec model')
        model = Word2Vec(sentences, size=192)
        model.save(WORD2VEC_MODEL_FILENAME)
    return model


def main():
    logging.basicConfig(level=logging.INFO)

    _, train_x_path, train_y_path, test_x_path, dict_txt_big = sys.argv

    if use_jieba:
        jieba.dt.tmp_dir = os.getcwd()
        jieba.load_userdict(dict_txt_big)

    logger.info('Loading data')
    train_x, train_y, test_x = read_data(
        train_x_path, train_y_path, test_x_path,
        dict_txt_big, use_jieba=use_jieba)
    print(train_x[-1], train_y[-1])

    word2vec_model = load_word2vec_model(train_x + test_x)

    wv = word2vec_model.wv

    keras_model = Sequential()

    weights = wv.vectors
    input_length = 40

    layer = Embedding(
        input_dim=weights.shape[0], output_dim=weights.shape[1],
        weights=[weights], trainable=True, input_length=input_length,
    )

    keras_model.add(layer)
    if use_rnn:
        keras_model.add(LSTM(256, dropout=0.5, recurrent_dropout=0.5))
        keras_model.add(Dense(256, activation='relu'))
        keras_model.add(Dropout(0.5))
    else:
        keras_model.add(Dropout(0.5))
        keras_model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
        keras_model.add(MaxPooling1D(3))
        keras_model.add(Flatten())
        keras_model.add(Dense(64, activation='relu'))
        keras_model.add(Dropout(0.5))
    keras_model.add(Dense(1, activation='sigmoid'))
    keras_model.compile(
        loss='binary_crossentropy', optimizer='nadam',
        metrics=['accuracy'])
    keras_model.summary()

    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(train_x)

    with open(os.path.join('tokenizer.pickle'), 'wb') as f:
        pickle.dump(tokenizer, f)

    source = tokenizer.texts_to_sequences(train_x)
    source = pad_sequences(source, maxlen=input_length)
    print(source.shape)
    print(source[-1])

    N = 110000
    # target = to_categorical(train_y)
    target = train_y

    checkpointer = ModelCheckpoint(
        os.path.join('Model-epoch_{epoch}-acc_{val_acc:.4f}.h5'),
        save_best_only=False)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', verbose=1)

    history = keras_model.fit(
        x=source[:N], y=target[:N], validation_data=(source[N:], target[N:]),
        epochs=5, callbacks=[checkpointer, lr_scheduler])

    network_type = 'rnn' if use_rnn else 'bow'
    with open('{}_history.bin'.format(network_type), 'wb') as f:
        pickle.dump(history.history, f)


if __name__ == '__main__':
    main()
