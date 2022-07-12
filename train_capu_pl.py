import json
import os
import time
import random
import argparse

import pickle
from transformers.optimization import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from tqdm import trange, tqdm
import numpy as np
import torch
import pytorch_lightning as pl
from model_pl import PL_HuyDangCapuModel
from torch.nn.utils import clip_grad_norm_
from transformers import RobertaModel
from transformers import RobertaModel, BertForTokenClassification
from transformers.file_utils import cached_path, hf_bucket_url
from importlib.machinery import SourceFileLoader
import os
from torch.utils.data import DataLoader, DistributedSampler
from dataset import CapuDataset, CAP_ID_TO_LABEL,CAP_LABEL_TO_ID,PUNC_ID_TO_LABEL,PUNCT_LABEL_TO_ID
from sklearn.metrics import classification_report


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_tag", default='capu')

    parser.add_argument("--max_len", default=256, type=int,
                        help="maximum length of input")

    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--warmup_ratio", type=float, default=-1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--theta", type=float,
                        help="weight of two tasks", default=0.25)
    parser.add_argument("--window_size", type=int,
                        default=100, help="size of the sliding window")
    parser.add_argument("--overlap", type=int, default=50,
                        help="overlap size of the two sliding windows")
    parser.add_argument("--threshold", type=int, default=5,
                        help="At least the number of times a possible relationship should appear in the training set "
                             "(should be greater than or equal to the threshold in the data preprocessing stage)")
    parser.add_argument("--local_rank", type=int, default=-
                        1, help="用于DistributedDataParallel")
    parser.add_argument("--max_grad_norm", type=float, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true",
                        help="whether to enable mixed precision")
    parser.add_argument("--not_save", action="store_true",
                        help="whether to save the model")
    parser.add_argument("--reload", action="store_true",
                        help="whether to reload data")
    parser.add_argument("--test_eval", action="store_true")
    args = parser.parse_args()
    return args


def download_tokenizer_files(cache_dir, model_name):
    resources = ['envibert_tokenizer.py', 'dict.txt', 'sentencepiece.bpe.model']
    for item in resources:
        if not os.path.exists(os.path.join(cache_dir, item)):
            tmp_file = hf_bucket_url(model_name, filename=item)
            tmp_file = cached_path(tmp_file, cache_dir=cache_dir)
            os.rename(tmp_file, os.path.join(cache_dir, item))

if __name__ == '__main__':
    args = args_parser()
 
    cache_dir = './cache'
    model_name = 'nguyenvulebinh/envibert'
    download_tokenizer_files(cache_dir,model_name)
    tokenizer = SourceFileLoader("envibert.tokenizer", os.path.join(cache_dir, 'envibert_tokenizer.py'))\
        .load_module().RobertaTokenizer( cache_dir)


    bert = RobertaModel.from_pretrained(model_name, cache_dir=cache_dir)
    print(bert)
    punct_class_weight = torch.Tensor([2.7295e-01, 5.1040e+00, 7.1929e+00, 6.9293e+02])
    cap_class_weight = torch.Tensor([0.6003, 2.9913])
    # punct_class_weight = torch.Tensor([1, 3,5, 10])
    # cap_class_weight = torch.Tensor([0.6003, 1.0])
    # punct_class_weight = None
    # cap_class_weight = None
    model = PL_HuyDangCapuModel(initialized_bert=bert, punctuation_class_weight=punct_class_weight,
                             capital_class_weight=cap_class_weight, kwargs = args)


    BATCH_SIZE = 64
    SHUFFLE = True

    train_dataset = CapuDataset('data_new/preprocessed/text_train.txt', 'data_new/preprocessed/labels_train.txt', tokenizer,
                                max_len=512, max_sample=5000, shuffle=True)
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, sampler=None,
                                  shuffle=SHUFFLE, drop_last=False)
    # eval_dataset = CapuDataset('data_new/preprocessed/text_dev.txt', 'data_new/preprocessed/labels_dev.txt', tokenizer,
    #                            max_len=512, max_sample=1000, shuffle=True)
    # eval_dataloader = DataLoader(eval_dataset, BATCH_SIZE, sampler=None,
    #                              shuffle=SHUFFLE, drop_last=False)

    trainer = pl.Trainer(accelerator="gpu", max_epochs=args.max_epochs)
    trainer.fit(model, train_dataloader)
