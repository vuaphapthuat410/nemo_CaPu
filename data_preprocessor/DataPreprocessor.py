import json
import unicodedata
from collections import Counter
import pickle
import re
from vncorenlp import VnCoreNLP
from nltk.tokenize import sent_tokenize
import random
from tqdm import tqdm
import numpy as np
import bs4
import string, os
from sklearn.metrics.pairwise import cosine_similarity
import pathlib
from typing import List



class VietnameseNewsPreprocessor:
    def __init__(self):
        currrent_dir = pathlib.Path(__file__).parent.resolve()
        with open(os.path.join(currrent_dir, 'remove_keyword.txt'),'r') as f:
            self.default_removal_list = f.read().splitlines()
        # self.punctuation_list = ['.',',','?','...','']
    #get text from html 
    def trankit_sentence_tokenize(self,text:str) -> List:
        sentences_dicts = self.trankit_pipeline(text)['sentences']
        sentence_ls = [a['text'] for a in sentences_dicts]
        return sentence_ls
    def del_html(self,html_text):
        soup = bs4.BeautifulSoup(html_text)
        return soup.get_text(' ')
    def del_css_js(self,text):
        text = re.sub('<.*?>', '', text) 
        text = re.sub('{.*}',' ',text)
        text = re.sub('(function).*{.*}',' ',text)
        text = re.sub('(function).*}',' ',text)
        return text
    def deEmojify(self,text):
        regrex_pattern = re.compile(pattern="["
                                            u"\U0001F600-\U0001F64F"  # emoticons
                                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                            "]+", flags=re.UNICODE)
        return regrex_pattern.sub(r'', text)
    def del_link(self,text):
        link = r'http[\S]*'
        text = re.sub(link, ' ', str(text))
        return text
    def normalize_text(self, text):
        # chuyen punctuation thành space
        # translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        # text = text.translate(translator)

        # text = text.replace(u'"', u'')
        # text = text.replace(u'️', u'')
        text = text.replace('🏻','')

        text.replace(");this.closest('table').remove();","")
        # text = re.sub(r"\s\s+", " ", full_text)
        text = re.sub(r"http\S+", "", text)
        text = re.sub('(\\r)*( )*(\\n)*( )*(\\r)*( )*(\\n)','.', text)
        text = re.sub(r"\.( )*(\.)+", '. ', text)
        text = re.sub('(\.(\s)+)+', '. ', text)
        text = re.sub('<[^<]+?>', '',text)
        text = text.replace('\u200c','')
        text = unicodedata.normalize('NFC', text)
        return text
    def default_remove(self,text:str = ''):
        for keyword in self.default_removal_list:
            text =  re.sub(keyword,'',text,flags=re.I)
        return text 
    def custom_remove(self,external_removal_list: List = [],text:str = ''):
        for keyword in external_removal_list:
            text =  re.sub(keyword,'',text,flags=re.I)
        return text

    """"
    Cleaning data, including remove css and js text, remove emoticon, remove frequently url link, replace escape sequence
    Can add extra keyword to remove from text. Support regular expression.
    """
    def data_cleaning(self,docs:list,external_removal_list = []):
        res = []
        for doc in tqdm(docs):
            
            doc = self.del_html(doc)
            #css and js
            doc = self.del_css_js(doc)
            #emoji
            doc = self.deEmojify(doc)
            #link
            doc = self.del_link(doc)
            #remove trash character
            doc = self.normalize_text(doc)
            #escape sequence
            doc = re.sub('\n',' ',doc)
            doc = re.sub('\t',' ',doc)
            doc = re.sub('\r',' ',doc)
            
            doc  = self.default_remove(doc)
            doc = self.custom_remove(external_removal_list=external_removal_list,text = doc)
            doc = doc.replace('\s+.','.').replace('\s+,',',')
            doc = re.sub('\s+',' ',doc)
            sentence_ls = sent_tokenize(doc)
            final_doc = ' '.join([sen for sen in sentence_ls if len(sen.split()) >= 5 ])
            final_doc = final_doc.replace(' .','.').replace('( ','(').replace(' )',')')
            if len(final_doc) == 0:
                res.append('')
            res.append(final_doc)
        return res

    """
    Segment a list of text with vncorenlp
    """
    def segmentation_vi(self,docs:list):
        for message in tqdm(docs):
            result = []
            try:
                message = message.replace('_',' ')
                segmented_message = ''
                segmented_sentences = self.annotator.tokenize(message)
                for sentence in segmented_sentences:
                    segmented_message += ' ' + ' '.join(sentence)   
                segmented_message = segmented_message.replace(' .','.').replace('( ','(').replace(' )',')')
                result.append(segmented_message) 
            except Exception:
                return []
        return result

    def random_infer(self,path,bunch):
        with open(path,'r',encoding='utf-8') as f:
            res = json.load(f)
        print('Original len: {}'.format(len(res)))
        random.shuffle(res)
        with open('check.json','w',encoding='utf-8') as f:
            json.dump(res[:bunch],f,ensure_ascii=False)

    def filter_by_cos_sim_tfidf(self,docs,threshold):

        tfidfs = self.vn_tfidf_builder.transform(docs).toarray()
        batch = 1000
        for i in range(len(docs) // batch + 1):
            stop = min(batch *(i+1), len(docs))
            print(batch * i, stop)
            similarities =  cosine_similarity(tfidfs[batch * i: stop],tfidfs)
            count =0

            for k in range(batch*i , stop - 1):
                for j in range(k+1, len(docs)):
                    if similarities[k - batch*i][j] > threshold :
                        docs[j] = ''
                        count+=1
            print(count)
        corpus = [doc for doc in docs if doc != '']
        print(len(corpus))
        return corpus
        
    def process(self,input_path:str,output_path:str,lang = 'vi',segment = False, feature = True,filter = True,check = False):
        
        with open(input_path,'r',encoding='utf-8') as f:
            corpus_dict = json.load(f)
        print('Input length: {}'.format(len(corpus_dict)))
        print('Cleaning...:')
        corpus_dict = self.data_cleaning(corpus_dict)
        print('Segment...:')
        if segment:
            if (lang == 'vi'):
                corpus_dict = self.segmentation_vi(corpus_dict,feature)
            else:
                pass
        if filter:
            print('Filtering...')

            #lang = 'vi
            with open('global/vn_tfidf_builder.pkl','rb') as f:
                vn_tfidf_builder = pickle.load(f)
            corpus_dict = self.filter_by_cos_sim_tfidf(corpus_dict,vn_tfidf_builder)
        with open(output_path,'w',encoding='utf-8') as f:
            json.dump(corpus_dict,f,ensure_ascii=False)
        if check == True:
            random.shuffle(corpus_dict)
            sample = 1000
            with open('check.json','w',encoding='utf-8') as f:
                json.dump(corpus_dict[:sample],f,ensure_ascii=False)


    

