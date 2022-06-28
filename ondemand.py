import random


path1 = '/home/huydang/project/nemo_capu/data_new/preprocessed/text_train.txt'
path2 = '/home/huydang/project/nemo_capu/data_new/preprocessed/labels_train.txt'

NUM = 700000

with open(path1) as f:
    ls = f.read().splitlines()[:NUM]



with open(path2) as f:
    label_ls = f.read().splitlines()[:NUM]

dict_ls = list()
for t,l in zip(ls,label_ls):
    temp_l = l.split()
    temp_l = [i[0] for i in temp_l]
    assert len(t.split()) == len(temp_l)
    dict_ls.append({'sentence': t, 'punctuation': ' '.join(temp_l)})
  
import json
print(dict_ls[:5])
with open('data_training_punctua.json','w') as f:
    json.dump(dict_ls,f,ensure_ascii=False)