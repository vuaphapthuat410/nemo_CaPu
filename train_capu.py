
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

from model import HuyDangCapuModel
from torch.nn.utils import clip_grad_norm_
from transformers import RobertaModel
from transformers import RobertaModel, BertForTokenClassification
from transformers.file_utils import cached_path, hf_bucket_url
from importlib.machinery import SourceFileLoader
import os
from torch.utils.data import DataLoader, DistributedSampler
from dataset import CapuDataset, CAP_ID_TO_LABEL,CAP_LABEL_TO_ID,PUNC_ID_TO_LABEL,PUNCT_LABEL_TO_ID

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
                        help="At least the number of times a possible relationship should appear in the training set (should be greater than or equal to the threshold in the data preprocessing stage)")
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




def train(args, train_dataloader,eval_dataloader,model):

    model.train()
    # if args.amp:
    scaler = GradScaler()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], "weight_decay":args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay":0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    if args.warmup_ratio > 0:
        num_training_steps = len(train_dataloader)*args.max_epochs
        warmup_steps = args.warmup_ratio*num_training_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, num_training_steps)
    if args.local_rank < 1:
        mid = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
    for epoch in range(args.max_epochs):
        if args.local_rank != -1:
            train_dataloader.sampler.set_epoch(epoch)

        tqdm_train_dataloader = tqdm(
            train_dataloader, desc="Training epoch:%d" % epoch)
        for i, batch in enumerate(tqdm_train_dataloader):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            input_ids, attention_mask, capital_labels, punctuation_labels = batch['input_ids'], batch['attention_mask'], batch['capital_labels'],\
                batch['punctuation_labels']

            input_ids, attention_mask, capital_labels, punctuation_labels = input_ids.type(torch.LongTensor).to(device), attention_mask.to(device), capital_labels.type(torch.LongTensor).to(device),\
                punctuation_labels.type(torch.LongTensor).to(device)

            print(punctuation_labels[0][:300])
            input()

            if args.amp:
                with autocast():
                    loss, (loss_t1, loss_t2) = model(
                        input_ids, attention_mask, capital_labels = capital_labels, punctuation_labels = punctuation_labels)
                scaler.scale(loss).backward()
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, (loss_t1, loss_t2) =  model(input_ids, attention_mask, capital_labels = capital_labels, punctuation_labels = punctuation_labels)
                loss.backward()
                if args.max_grad_norm > 0:
                    clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            lr = optimizer.param_groups[0]['lr']
            named_parameters = [
                (n, p) for n, p in model.named_parameters() if not p.grad is None]
            grad_norm = torch.norm(torch.stack(
                [torch.norm(p.grad) for n, p in named_parameters])).item()
            if args.warmup_ratio > 0:
                scheduler.step()
            tqdm_dict = dict()
            tqdm_dict["Loss"] = "{:.5f}".format(loss.item())
            tqdm_dict["Capital_loss"] = "{:.5f}".format(loss_t1)
            tqdm_dict["Punctuation_loss"] = "{:.5f}".format(loss_t2)
            tqdm_train_dataloader.set_postfix(tqdm_dict)

        if args.local_rank in [-1, 0] and not args.not_save:
            if hasattr(model, 'module'):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()

            save_dir = './checkpoints/%s/%s/' % (args.dataset_tag, mid)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                pickle.dump(args, open(save_dir+'args', 'wb'))
            save_path = save_dir+"checkpoint_%d.ckpt" % epoch
            if epoch % 5 == 0 or epoch == args.max_epochs - 1:
                torch.save(model_state_dict, save_path)
                print("model saved at:", save_path)


        tqdm_eval_dataloader = tqdm(
            eval_dataloader, desc="Evaluating epoch:%d" % epoch)
        for i, batch in enumerate(tqdm_eval_dataloader):

            torch.cuda.empty_cache()

            input_ids, attention_mask, capital_labels, punctuation_labels = batch['input_ids'], batch['attention_mask'], batch['capital_labels'],\
                batch['punctuation_labels']
            input_ids, attention_mask, capital_labels, punctuation_labels = input_ids.type(torch.LongTensor).to(device), attention_mask.to(device), capital_labels.type(torch.LongTensor).to(device),\
                punctuation_labels.type(torch.LongTensor).to(device)

            with torch.no_grad():
                if args.amp:
                    with autocast():
                        loss, (loss_t1, loss_t2) = model(
                            input_ids, attention_mask, capital_labels = capital_labels, punctuation_labels = punctuation_labels)
                else:
                    loss, (loss_t1, loss_t2), capital_pred, punctuation_pred =  model(input_ids, attention_mask, capital_labels = capital_labels, punctuation_labels = punctuation_labels,return_idx = True)
                    if i == len(tqdm_eval_dataloader) - 1:
                        torch.save(punctuation_pred,'save_obj/punctuation_pred.pt')
                        torch.save(capital_pred, 'save_obj/capital_pred.pt')
            tqdm_dict = {}
            tqdm_dict["Eval loss"] = "{:.5f}".format(loss.item())
            tqdm_dict["Eval Capital_loss"] = "{:.5f}".format(loss_t1)
            tqdm_dict["Eval Punctuation_loss"] = "{:.5f}".format(loss_t2)
            tqdm_eval_dataloader.set_postfix(tqdm_dict)

        if args.local_rank != -1:
            torch.distributed.barrier()




def download_tokenizer_files(cache_dir, model_name):
    resources = ['envibert_tokenizer.py', 'dict.txt', 'sentencepiece.bpe.model']
    for item in resources:
        if not os.path.exists(os.path.join(cache_dir, item)):
            tmp_file = hf_bucket_url(model_name, filename=item)
            tmp_file = cached_path(tmp_file, cache_dir=cache_dir)
            os.rename(tmp_file, os.path.join(cache_dir, item))

if __name__ == '__main__':
    args = args_parser()
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend='nccl')

    cache_dir = './cache'
    model_name = 'nguyenvulebinh/envibert'
    download_tokenizer_files(cache_dir,model_name)
    tokenizer = SourceFileLoader("envibert.tokenizer", os.path.join(cache_dir, 'envibert_tokenizer.py')).load_module().RobertaTokenizer( cache_dir)


    bert = RobertaModel.from_pretrained(model_name, cache_dir=cache_dir)
    punct_class_weight = torch.Tensor([2.7295e-01, 5.1040e+00, 7.1929e+00, 6.9293e+02])
    cap_class_weight = torch.Tensor([0.6003, 2.9913])
    model = HuyDangCapuModel(initialized_bert=bert,punctuation_class_weight=punct_class_weight, capital_class_weight=cap_class_weight)

    path = '/home/huydang/project/nemo_capu/checkpoints/training_500k/2022_06_06_08_52_53/checkpoint_9.ckpt'
    model.load_state_dict(torch.load(path))
    BATCH_SIZE = 1
    SHUFFLE = True

    train_dataset = CapuDataset('data/preprocessed/text_train.txt', 'data/preprocessed/labels_train.txt', tokenizer, max_len=512, max_sample=5000)
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, sampler=None,
                                  shuffle=SHUFFLE, drop_last=False)
    eval_dataset = CapuDataset('data/preprocessed/text_dev.txt', 'data/preprocessed/labels_dev.txt', tokenizer, max_len=512, max_sample=2000)
    eval_dataloader = DataLoader(eval_dataset, BATCH_SIZE, sampler=None,
                                 shuffle=SHUFFLE, drop_last=False)

    train(args, train_dataloader, eval_dataloader, model)
