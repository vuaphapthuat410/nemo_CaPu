import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import AutoModel, AutoTokenizer, RobertaModel
from dataset import CapuDataset

from typing import List, Optional, Tuple, Union
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from dataset import CAP_ID_TO_LABEL, CAP_LABEL_TO_ID, PUNC_ID_TO_LABEL, PUNCT_LABEL_TO_ID
from importlib.machinery import SourceFileLoader
import os

cache_dir = './cache'
model_name = 'nguyenvulebinh/envibert'
default_tokenizer = SourceFileLoader("envibert.tokenizer",
                             os.path.join(cache_dir, 'envibert_tokenizer.py')).load_module().RobertaTokenizer(cache_dir)

class HuyDangCapuModel(nn.Module):
    def __init__(self, pretrained_name = None, initialized_bert = None,punctuation_class_weight = None, capital_class_weight = None):
        super(HuyDangCapuModel, self).__init__()
        if not initialized_bert:
            self.bert = AutoModel.from_pretrained(pretrained_name)
        else:
            self.bert = initialized_bert

        self.config = self.bert.config

        self.num_token_labels = 4
        self.dropout = nn.Dropout(0.1)
        self.punctuation_classifier = nn.Linear(self.bert.config.hidden_size, self.num_token_labels)
        self.capital_classifier = nn.Linear(self.bert.config.hidden_size, 2)
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

        self.theta = 0.5

    def forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        capital_labels: Optional[torch.LongTensor] = None,
        punctuation_labels: Optional[torch.LongTensor] = None,return_idx = False):

        capital_loss = 0
        punctuation_loss = 0

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)

        sequence_output = outputs[0]
        sequence_output_for_token_classifier = self.dropout(sequence_output)
        punctuation_classifier_logits = self.punctuation_classifier(sequence_output_for_token_classifier)
        capital_classifier_logits = self.capital_classifier(sequence_output_for_token_classifier)

        if capital_labels is None and punctuation_labels is None:

            capital_idxs = torch.argmax(capital_classifier_logits, dim=-1).squeeze(-1)
            punctuation_ids = torch.argmax(punctuation_classifier_logits, dim=-1).squeeze(-1)

            return capital_idxs, punctuation_ids

        else:
            # if self.punctuation_class_weight is not None:
            #     punctuation_loss_fct = CrossEntropyLoss(weight=self.punctuation_class_weight, reduction='mean')
            # else:
            #     punctuation_loss_fct = CrossEntropyLoss()
            #
            # if self.capital_class_weight is not None:
            #     capital_loss_fct = CrossEntropyLoss(weight=self.capital_class_weight, reduction='mean')
            # else:
            #     capital_loss_fct = CrossEntropyLoss()

            if capital_labels is not None:

                capital_loss = self.capital_loss_fct(capital_classifier_logits.view(-1,2),capital_labels.view(-1))

            if punctuation_labels is not None:
                punctuation_loss = self.punctuation_loss_fct(punctuation_classifier_logits.view(-1,4),punctuation_labels.view(-1))

            total_loss = self.theta * capital_loss + (1 - self.theta) * punctuation_loss
            if not return_idx:
                return total_loss, (capital_loss.item(), punctuation_loss.item())
            else:
                capital_idxs = torch.argmax(capital_classifier_logits, dim=-1).squeeze(-1)
                punctuation_ids = torch.argmax(punctuation_classifier_logits, dim=-1).squeeze(-1)
                return total_loss, (capital_loss.item(), punctuation_loss.item()), capital_idxs, punctuation_ids

    def _map_valid_id(self,idxs_batch,tokens_batch,queries_batch,mapper):
        labels_batch = []
        for q in range(idxs_batch.shape[0]):
            t_cap_labels = []

            for token_ind in range(1, idxs_batch.shape[1]):
                if token_ind - 1 >= len(tokens_batch[q]):
                    break
                if tokens_batch[q][token_ind - 1].startswith('▁'):
                    t_cap_labels.append(mapper[idxs_batch[q][token_ind].item()])

            assert len(t_cap_labels) == len(queries_batch[q])
            labels_batch.append(t_cap_labels)


        return labels_batch
    def infer(self,text_ls =  List[str],initialized_tokenizer = default_tokenizer,batch_size = None):
        INFER_BATCH = batch_size if batch_size is not None else 32
        INFER_BATCH = min(INFER_BATCH,len(text_ls))

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        infer_dataset = CapuDataset(tokenizer=initialized_tokenizer,max_len=512,infer=True,infer_text_ls=text_ls)

        infer_dataloader = DataLoader(infer_dataset, INFER_BATCH, sampler=None,
                                      shuffle=False, drop_last=True)
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

                queries_batch = text_ls[i*INFER_BATCH: i * INFER_BATCH  + capital_idxs.shape[0]]
                tokens_batch = [initialized_tokenizer.tokenize(q) for q in queries_batch]
                queries_batch = [q.split() for q in queries_batch]
                cap_labels_batch = self._map_valid_id(capital_idxs,tokens_batch,queries_batch,CAP_ID_TO_LABEL)
                punct_labels_batch = self._map_valid_id(punctuation_ids,tokens_batch,queries_batch,PUNC_ID_TO_LABEL)

                for ind,query in enumerate(queries_batch):
                    w_ls = []
                    for w,cap, punct in zip(query,cap_labels_batch[ind],punct_labels_batch[ind]):
                        t_w = w.title() if cap == 'U' else w.lower() + punct
                        w_ls.append(t_w)

                    text_ind = ' '.join(w_ls)
                    res_batch.append(text_ind)

            return res_batch




if __name__ == '__main__':

    path = '/home/huydang/project/nemo_capu/checkpoints/training_500k_fix_weight/2022_06_07_17_46_37/checkpoint_9.ckpt'

    bert = RobertaModel.from_pretrained(model_name, cache_dir=cache_dir)
    punct_class_weight = torch.Tensor([2.7295e-01, 5.1040e+00, 7.1929e+00, 6.9293e+02])
    cap_class_weight = torch.Tensor([0.6003, 2.9913])
    model = HuyDangCapuModel(None, bert, punctuation_class_weight=punct_class_weight,capital_class_weight=cap_class_weight)
    model.load_state_dict(torch.load(path))

    sentence_ls = ['nếu điều trị ngoại trú sẽ không được hưởng bảo hiểm y tế nếu con chị đi khám tại bệnh viện tuyến huyện ở hà nội sẽ được hưởng 80 chi phí khám bệnh chữa bệnh','bạn tên là gì']


    print(model.infer(text_ls = sentence_ls ,batch_size = 32))