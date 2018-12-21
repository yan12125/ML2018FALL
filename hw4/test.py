import os
import pickle
import sys

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

use_jieba = True


def main():
    _, test_x_path, dict_txt_big, output_file = sys.argv
    if use_jieba:
        import jieba
        jieba.load_userdict(dict_txt_big)

    input_length = 40

    model_name = os.getenv('MODEL_NAME', 'Model.h5')
    raw_score = int(os.getenv('RAW_SCORE', 0))
    model = load_model(model_name)
    model.summary()

    with open('tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)

    test_x = []
    with open(test_x_path) as f:
        for row in f:
            row = row.split(',', maxsplit=1)
            if row[0] == 'id':
                continue
            if use_jieba:
                data_item_x = list(jieba.cut(row[1]))
            else:
                data_item_x = list(row[1])
            test_x.append(data_item_x)
    test_x = tokenizer.texts_to_sequences(test_x)
    test_x = pad_sequences(test_x, maxlen=input_length)

    prediction = model.predict(test_x)

    with open(output_file, 'w') as f:
        print('id,label', file=f)
        for idx, output in enumerate(prediction):
            if raw_score:
                result = output[0]
            else:
                result = 1 if output[0] > 0.5 else 0
            print(f'{idx},{result}', file=f)


if __name__ == '__main__':
    main()
