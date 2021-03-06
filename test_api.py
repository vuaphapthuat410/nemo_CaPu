import traceback

from fastapi import FastAPI
import uvicorn

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import AutoModel, AutoTokenizer, RobertaModel
from dataset import CapuDataset

from underthesea import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from typing import List, Optional, Tuple, Union
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from dataset import CAP_ID_TO_LABEL, CAP_LABEL_TO_ID, PUNC_ID_TO_LABEL, PUNCT_LABEL_TO_ID
from importlib.machinery import SourceFileLoader
from nemo.collections.nlp.modules.common import TokenClassifier

import os
import json
import time

# prerequisite for envibert
from nemo.collections.common.losses import AggregatorLoss as nemo_AggregatorLoss, CrossEntropyLoss as nemo_CrossEntropyLoss
cache_dir = './cache'
model_name = 'nguyenvulebinh/envibert'
default_tokenizer = SourceFileLoader("envibert.tokenizer",
                    os.path.join(cache_dir, 'envibert_tokenizer.py')).load_module().RobertaTokenizer(cache_dir)

class HuyDangCapuModel(nn.Module):

    def __init__(self, pretrained_name=None, initialized_bert=None, punctuation_class_weight=None,
                 capital_class_weight=None):
        super(HuyDangCapuModel, self).__init__()
        if not initialized_bert:
            self.bert = AutoModel.from_pretrained(pretrained_name)
        else:
            self.bert = initialized_bert

        self.config = self.bert.config

        self.num_token_labels = 4
        self.dropout = nn.Dropout(0.1)

        self.punctuation_classifier = TokenClassifier(
            hidden_size=self.bert.config.hidden_size,
            num_classes=self.num_token_labels,
            activation='relu',
            log_softmax=False,
            dropout=0.1,
            num_layers=1,
            use_transformer_init=True,
        )

        self.capital_classifier = TokenClassifier(
            hidden_size=self.bert.config.hidden_size,
            num_classes=2,
            activation='relu',
            log_softmax=False,
            dropout=0.1,
            num_layers=1,
            use_transformer_init=True,
        )
        self.punctuation_class_weight = punctuation_class_weight
        self.capital_class_weight = capital_class_weight

        if self.punctuation_class_weight is not None:
            self.punctuation_loss_fct = CrossEntropyLoss(weight=self.punctuation_class_weight, reduction='mean')
        else:
            self.punctuation_loss_fct = CrossEntropyLoss()

        if self.capital_class_weight is not None:
            self.capital_loss_fct = CrossEntropyLoss(weight=self.capital_class_weight, reduction='mean')
        else:
            self.capital_loss_fct = CrossEntropyLoss()

        self.agg_loss = nemo_AggregatorLoss(num_inputs=2)

        self.theta = 0.5

    def forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        capital_labels: Optional[torch.LongTensor] = None,
        punctuation_labels: Optional[torch.LongTensor] = None,
        loss_mask: Optional[torch.BoolTensor] = None, return_idx: bool = False):

        capital_loss = 0
        punctuation_loss = 0

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)

        sequence_output_for_token_classifier = outputs[0]
        sequence_output_for_token_classifier = self.dropout(sequence_output_for_token_classifier)
        punctuation_classifier_logits = self.punctuation_classifier(hidden_states=sequence_output_for_token_classifier)
        capital_classifier_logits = self.capital_classifier(hidden_states=sequence_output_for_token_classifier)

        # Equal to infer mode
        if capital_labels is None and punctuation_labels is None:

            capital_idxs = torch.argmax(capital_classifier_logits, dim=-1).squeeze(-1)
            punctuation_ids = torch.argmax(punctuation_classifier_logits, dim=-1).squeeze(-1)

            return capital_idxs, punctuation_ids

        else:

            if capital_labels is not None:

                capital_loss = self.capital_loss_fct(capital_classifier_logits[loss_mask],
                                                     capital_labels[loss_mask])
            if punctuation_labels is not None:
                punctuation_loss = self.punctuation_loss_fct(punctuation_classifier_logits[loss_mask],
                                                             punctuation_labels[loss_mask])

            total_loss = self.agg_loss(loss_1=punctuation_loss, loss_2=capital_loss)

            if not return_idx:
                return total_loss, (capital_loss.item(), punctuation_loss.item())
            else:
                capital_idxs = torch.argmax(capital_classifier_logits, dim=-1)
                punctuation_ids = torch.argmax(punctuation_classifier_logits, dim=-1)
                return total_loss, (capital_loss.item(), punctuation_loss.item()), capital_idxs, punctuation_ids

    def _map_valid_id(self, idxs_batch, tokens_batch, queries_batch, mapper):

        labels_batch = []
        for q in range(idxs_batch.shape[0]):
            t_cap_labels = []

            for token_ind in range(1, idxs_batch.shape[1]):
                if token_ind - 1 >= len(tokens_batch[q]):
                    break
                if tokens_batch[q][token_ind - 1].startswith('???'):
                    t_cap_labels.append(mapper[idxs_batch[q][token_ind].item()])

            assert len(t_cap_labels) == len(queries_batch[q])
            labels_batch.append(t_cap_labels)


        return labels_batch
    def infer(self,text_ls =  List[str],initialized_tokenizer = default_tokenizer,batch_size = None):
        INFER_BATCH = batch_size if batch_size is not None else 32
        INFER_BATCH = min(INFER_BATCH,len(text_ls))

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        infer_dataset = CapuDataset(tokenizer=initialized_tokenizer, max_len=512, infer=True, infer_text_ls=text_ls)

        infer_dataloader = DataLoader(infer_dataset, INFER_BATCH, sampler=None,
                                      shuffle=False, drop_last=False)
        self.eval()
        tqdm_train_dataloader = tqdm(infer_dataloader,  ncols=100)
        self.to(device)
        res_batch = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm_train_dataloader):
                torch.cuda.empty_cache()

                input_ids, attention_mask,   = batch['input_ids'], batch['attention_mask']
                input_ids, attention_mask,   = input_ids.type(torch.LongTensor).to(
                    device), attention_mask.to(device)

                capital_idxs, punctuation_ids = self.forward(
                    input_ids, attention_mask, capital_labels=None, punctuation_labels=None)

                queries_batch = text_ls[i*INFER_BATCH: i * INFER_BATCH + capital_idxs.shape[0]]
                tokens_batch = [initialized_tokenizer.tokenize(q) for q in queries_batch]
                queries_batch = [q.split() for q in queries_batch]
                cap_labels_batch = self._map_valid_id(capital_idxs, tokens_batch, queries_batch, CAP_ID_TO_LABEL)
                # print('cap')
                # print(cap_labels_batch)
                punct_labels_batch = self._map_valid_id(punctuation_ids, tokens_batch, queries_batch, PUNC_ID_TO_LABEL)
                # print('punct')
                # print(punct_labels_batch)
                for ind,query in enumerate(queries_batch):
                    w_ls = []
                    for w,cap, punct in zip(query, cap_labels_batch[ind], punct_labels_batch[ind]):
                        t_w = w.title() if cap == 'U' else w.lower() + punct
                        w_ls.append(t_w)

                    text_ind = ' '.join(w_ls)
                    res_batch.append(text_ind)

            return res_batch


def remove_punctuation(test_str):
    punc = ''';,.?'''
 
    # Removing punctuations in string
    # Using loop + punctuation string
    for ele in test_str:
        if ele in punc:
            test_str = test_str.replace(ele, "")
    
    return test_str

#init app
app = FastAPI()

path = '/home/huydang/project/nemo_capu/checkpoints/training_new128_5000k_from_ep10_formal_weighted/2022_06_20_07_53_10/checkpoint_35.ckpt'

bert = RobertaModel.from_pretrained(model_name, cache_dir=cache_dir)
punct_class_weight = torch.Tensor([2.7295e-01, 5.1040e+00, 7.1929e+00, 6.9293e+02])
cap_class_weight = torch.Tensor([0.6003, 2.9913])
# punct_class_weight = None
# cap_class_weight = None
model = HuyDangCapuModel(None, bert, punctuation_class_weight=punct_class_weight, capital_class_weight=cap_class_weight)
model.load_state_dict(torch.load(path))

@app.get("/restore")
async def restore(text: str):
  start = time.time()

  res_str = model.infer(text_ls = [text] ,batch_size = 1)

  end = time.time()

  return {"time": end - start, 'before': text, 'result': res_str}

@app.get("/rm_then_rs")
async def rm_then_rs(text: str):
  start = time.time()

  normal_t = remove_punctuation(text).lower()
  res_str = model.infer(text_ls = [normal_t] ,batch_size = 1)

  end = time.time()

  origin_t_segs = word_tokenize(text)
  res_t_segs = word_tokenize(res_str[0])

  print(origin_t_segs)
  print(res_t_segs)

  bleu_score = sentence_bleu([origin_t_segs], res_t_segs, smoothing_function=SmoothingFunction().method7)

  return {"time": end - start, 'before': text, 'result': res_str[0], 'bleu_score': bleu_score}

if __name__ == '__main__':

    uvicorn.run("test_api:app", host="0.0.0.0", port=8701)

    # sentence_ls = ['','h??m qua tr???i m??a t??i kh??ng ??i l??m tr?????ng ph??ng g???i ??i???n nh???n tin t??i c??ng m???c k???','b???n t??n l?? g??','??ng nguy???n v??n  linh - tr?????ng ph??ng th???c t???p v???a m???i ???????c t??ng l????ng','??i khi n??o']

    # print(sentence_ls)
    # print(model.infer(text_ls = sentence_ls ,batch_size = 32))

    # with open('test.txt') as f:
    #     test_str = f.read()
    
    # normal_t = remove_punctuation(test_str).lower()
    # print(normal_t)
    # print(model.infer(text_ls = [normal_t] ,batch_size = 1))\

