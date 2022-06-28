

import pytorch_lightning as pl
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import List, Optional, Tuple, Union
import torch
from sklearn.metrics import classification_report


class PL_HuyDangCapuModel(pl.LightningModule):
    def __init__(self, pretrained_name=None, initialized_bert=None, punctuation_class_weight=None,
                 capital_class_weight=None, kwargs=None):
        super(HuyDangCapuModel, self).__init__()
        if not initialized_bert:
            self.bert = AutoModel.from_pretrained(pretrained_name)
        else:
            self.bert = initialized_bert

        self.config = self.bert.config

        self.num_token_labels = 4
        self.dropout = nn.Dropout(0.1)
        self.kwargs = kwargs

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
        # capital_labels: Optional[torch.LongTensor] = None,
        # punctuation_labels: Optional[torch.LongTensor] = None,
        # loss_mask: Optional[torch.BoolTensor] = None, return_idx: bool = False
        ):

        capital_loss = 0
        punctuation_loss = 0

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)

        sequence_output_for_token_classifier = outputs[0]
        sequence_output_for_token_classifier = self.dropout(sequence_output_for_token_classifier)
        punctuation_classifier_logits = self.punctuation_classifier(hidden_states=sequence_output_for_token_classifier)
        capital_classifier_logits = self.capital_classifier(hidden_states=sequence_output_for_token_classifier)
        return punctuation_classifier_logits, capital_classifier_logits
        # Equal to infer mode
 


        
        else:

           

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
                if tokens_batch[q][token_ind - 1].startswith('▁'):
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

    
    def compute_punctuation_metrics(punctuation_ids: Optional[torch.LongTensor] = None, punctuation_labels: Optional[torch.LongTensor] = None):
        punctuation_ids = punctuation_ids.cpu().numpy()
        punctuation_labels = punctuation_labels.cpu().numpy()
        punct_cls_rp = classification_report(punctuation_ids, punctuation_labels, output_dict=True, zero_division=0)

        return punct_cls_rp['accuracy'], punct_cls_rp['0']['f1-score'], punct_cls_rp['1']['f1-score'], punct_cls_rp['2']['f1-score'], punct_cls_rp['3']['f1-score']

    def compute_capital_metrics(capital_idxs: Optional[torch.LongTensor] = None, capital_labels: Optional[torch.LongTensor] = None):
        capital_idxs = capital_idxs.cpu().numpy()
        capital_labels = capital_labels.cpu().numpy()
        capi_cls_rp = classification_report(capital_idxs, capital_labels, output_dict=True, zero_division=0)

        return capi_cls_rp['accuracy'], capi_cls_rp['0']['f1-score'], capi_cls_rp['1']['f1-score']
        
    def training_step(self, bacth, batch_idx):
        input_ids, attention_mask, capital_labels, punctuation_labels, loss_mask, subtoken_mask = \
        batch['input_ids'], batch['attention_mask'], batch['capital_labels'],\
        batch['punctuation_labels'], batch['loss_mask'], batch['subtoken_mask']
        punctuation_classifier_logits, capital_classifier_logits = model(input_ids,attention_mask)
      
        capital_loss = self.capital_loss_fct(capital_classifier_logits[loss_mask],
                                            capital_labels[loss_mask])

        punctuation_loss = self.punctuation_loss_fct(punctuation_classifier_logits[loss_mask],
                                                    punctuation_labels[loss_mask])
        total_loss = self.agg_loss(loss_1=punctuation_loss, loss_2=capital_loss)
     
        self.log('Train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        punctuation_ids = torch.argmax(punctuation_classifier_logits, dim=-1).squeeze(-1)
        capital_idxs = torch.argmax(capital_classifier_logits, dim=-1).squeeze(-1)
        


    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], "weight_decay":self.kwargs.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], "weight_decay":0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.kwargs.lr)
        return optimizer



    