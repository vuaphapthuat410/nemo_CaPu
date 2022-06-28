import random


def random_show(path, number = 100):
    with open(path) as f:
        ls = f.read().splitlines()
    # random.shuffle(ls)
    ls = ls[:number]
    with open('sample.txt', 'w') as f:
        f.write('\n'.join(ls))

path = '/home/huydang/project/nemo_capu/data_new/preprocessed/text_train.txt'

random_show(path)
