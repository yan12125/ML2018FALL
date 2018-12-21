import jieba


def parse_sentences(data_x, use_jieba):
    ret = []
    with open(data_x) as f_x:
        for row_x in f_x:
            row_x = row_x.strip().split(',', maxsplit=1)
            if row_x[0] == 'id':
                continue
            data_item_x = row_x[1]
            if use_jieba:
                data_item_x = list(jieba.cut(data_item_x))
            else:
                data_item_x = list(data_item_x)
            ret.append(data_item_x)
    return ret
