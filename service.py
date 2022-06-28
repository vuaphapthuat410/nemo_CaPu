
import traceback

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import HuyDangCapuModel
from transformers import RobertaModel
import re
from pydantic import BaseModel
from importlib.machinery import SourceFileLoader
import os
import torch
class PostAsrTextInput(BaseModel):
    text: str = ''

class PostAsrTextOutput(BaseModel):
    status: int = 1
    message: str = ""
    result: dict = {}



app = FastAPI()

app.add_middleware( 
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


cache_dir = './cache'
model_name = 'nguyenvulebinh/envibert'
default_tokenizer = SourceFileLoader("envibert.tokenizer",
                    os.path.join(cache_dir, 'envibert_tokenizer.py')).load_module().RobertaTokenizer(cache_dir)

path = '/home/huydang/project/nemo_capu/checkpoints/training_new128_5000k_formal_weight/2022_06_17_03_02_06/checkpoint_9.ckpt'

bert = RobertaModel.from_pretrained(model_name, cache_dir=cache_dir)
punct_class_weight = torch.Tensor([2.7295e-01, 5.1040e+00, 7.1929e+00, 6.9293e+02])
cap_class_weight = torch.Tensor([0.6003, 2.9913])
model = HuyDangCapuModel(None, bert, punctuation_class_weight=punct_class_weight, capital_class_weight=cap_class_weight)
model.load_state_dict(torch.load(path))


def  wrap_infer(text):
    try:
        text = model.infer([text],batch_size=32)[0]
        return {'message': 'fail','text': text}
    except Exception:
        return {'message': 'fail','text': ''}

@app.post("/post_asr_norm/vi/fortest/vietnamese_capu_nemo_ver")
def vietnamese_capu_nemo_ver(text:str):
    status = 1
    try:          
   
        raw = re.sub(r"[,:.!;?]", '', text)
        raw = raw.lower()
        results = wrap_infer(raw)

        message = results['message']
        result = {'text': results['text'],'message': 'success','raw':raw}
    except: 
        traceback.print_exc()
        result = {'text': '','message': 'fail'}
    message = result['message']
    del result['message']
    return PostAsrTextOutput(status=status, message=message, result=result)

@app.post("/post_asr_norm/vi/vietnamese_capu_nemo_ver")
def vietnamese_capu_nemo_ver(item: PostAsrTextInput):
    status = 1
    try:          
        text = item.text      
        raw = re.sub(r"[,:.!;?]", '', text)
        raw = raw.lower()
        results = wrap_infer(raw)

        message = results['message']
        result = {'text': results['text'],'message': 'success','raw':raw}
    except: 
        traceback.print_exc()
        result = {'text': '','message': 'fail'}
    message = result['message']
    del result['message']
    return PostAsrTextOutput(status=status, message=message, result=result)