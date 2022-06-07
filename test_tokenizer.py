from transformers import RobertaModel
from transformers.file_utils import cached_path, hf_bucket_url
from importlib.machinery import SourceFileLoader
import os

cache_dir='./cache'
model_name='nguyenvulebinh/envibert'

def download_tokenizer_files():
  resources = ['envibert_tokenizer.py', 'dict.txt', 'sentencepiece.bpe.model']
  for item in resources:
    if not os.path.exists(os.path.join(cache_dir, item)):
      tmp_file = hf_bucket_url(model_name, filename=item)
      tmp_file = cached_path(tmp_file,cache_dir=cache_dir)
      os.rename(tmp_file, os.path.join(cache_dir, item))

download_tokenizer_files()
tokenizer = SourceFileLoader("envibert.tokenizer", os.path.join(cache_dir,'envibert_tokenizer.py')).load_module().RobertaTokenizer(cache_dir)
model = RobertaModel.from_pretrained(model_name,cache_dir=cache_dir)

# Encode text
# with open('error_text.txt',encoding='utf8') as f:
#   text = f.read()
# text = text.replace('‌','')
text = ' Sau khi nhận " hàng ", An mua ve tàu hoả SE6 đi Nha Trang,'
print(tokenizer.tokenize(text))
# print(text)
# input()
# label = 'OU ,O OU OU OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO .U OU OO OO OO OO ,O OO OO OO OO OO OO OO OO OO OO OO .O OU OO ,O OO OO OO OO OO OO OO OO OO OO OO OO ,O OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO ,O OO OO OU OU OO OO OO OU OU OO OO OO OU OU OO OO OO OO OO OO OO OO OO OO OO OO ,O .O OO OO OO OO OO OO OO OO OO OO OO OO OO .O OU OU OU OO ,O OO OO ,O OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OO OU OU .O OU ,O OO OO OO OO OO OO OO OO OO OO .O OU OO OO OO OO OO ,O OO OO OO OU OO OO OO OO OO OO OO OO OO ,O OO OO OO OO OO OO OU OU OU OU OO OO OO OO OO OO OO OU OO OO OU ,U OU OO OO OO OU OU OO OO .O OU OO OO OO OO OO OO OO OO OO OO OU ,U OU ,U OU ,U OU ,U OU ,U OU ,U OU ,U OU ,U OU ,U OU ,U OU ,U OU OU OO OO OO .O'
# count = 0
# word_ls = text.split()
# print(tokenizer(text))
# print(len(tokenizer.tokenize(text)))
# for ind, token_i in enumerate(tokenizer.tokenize(text)):

  # if token_i.startswith('▁'):
#   print(word_ls[count], '----', token_i)
#   count += 1
# #
# print(count)
# print(len(label.split()))
# # print(res)
# # print(text_ids)
# #
# # res = tokenizer(text_input2, return_tensors='pt')
# # text_ids = res.input_ids
# #
# #
# #

# # for tk in tokenizer.tokenize(text_input2):
# print(tokenizer.convert_tokens_to_ids('_xin'))
# print(tokenizer.convert_tokens_to_ids('xin'))


