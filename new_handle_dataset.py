import json
# from preprocessing import *
from typing import List
import string
import os
import random
import re
import numpy as np
from nltk.tokenize import sent_tokenize
from data_preprocessor import VietnameseNewsPreprocessor
label_dict = {'O': 0,'PERIOD':1, 'COMMA':2, 'COLON':3, 'QMARK':4}
PUNCTUATIONS = ['O','PERIOD', 'COMMA', 'QMARK', 'PERIOD', 'COLON','COMMA']
PUNCTUATION_SYM = ['','.',',','?','!',':',';']
# EOS_TOKENS = ['PERIOD','QMARK','EXCLAM']
EOS_TOKENS = ['PERIOD','QMARK']

folder_data_path = '/home/linhnguyen/news'

# category_list = ['Chính trị','Đối ngoại','Giáo dục','Pháp luật','Quân sự','Văn hóa','Xã hội']
category_list =  os.listdir(folder_data_path)


def normalize_text(text):

    text = text.replace('🏻', '')
    text = text.replace('“','"')
    full_text_clean = text
    full_text_clean.replace(");this.closest('table').remove();", "")
    full_text_clean = re.sub(
        '(Thứ .{2,4}|Chủ nhật),( ngày)? \d{1,2}\/\d{1,2}\/\d{4}( \d{1,2}:\d{1,2})?( AM| PM)?( \(GMT.{1,3}\))?', '',
        full_text_clean)



    full_text_clean = re.sub('\(.*(Ảnh|Nguồn).*?\)', '', full_text_clean)
    full_text_clean = re.sub('\d{1,2} (giờ|phút) trước', '', full_text_clean)
    full_text_clean = re.sub(r"http\S+", "", full_text_clean)
    full_text_clean = re.sub('(\\r)*( )*(\\n)*( )*(\\r)*( )*(\\n)', '.', full_text_clean)
    full_text_clean = re.sub(r"\.( )*(\.)+", '. ', full_text_clean)
    full_text_clean = re.sub('\.(?!\d)', '. ', full_text_clean)
    full_text_clean = re.sub('(\.(\s)+)+', '. ', full_text_clean)
    full_text_clean = re.sub('<[^<]+?>', '', full_text_clean)
    full_text_clean = re.sub('\d{1,2}:\d{2}( )?\d{1,2}\/\d{1,2}\/\d{4}', '', full_text_clean)
    full_text_clean = re.sub("Ảnh(:)?(Getty)?", "", full_text_clean)
    full_text_clean = full_text_clean.replace("Read more about:", "").replace("Read more", "").replace("Advertising",
                                                                                                       "").replace(
        "bookmark border.", "").replace(
        'the latest tech news, global tech news daily, tech news today, startups, usa tech, asia tech, china tech, eu tech, global tech, in-depth electronics reviews, 24h tech news, 24h tech news, top mobile apps, tech news daily, gaming hardware, big tech news, useful technology tips, expert interviews, reporting on the business of technology, venture capital funding, programing language',
        "").replace('Live updates:', '').replace('-VNExplorer', '').replace('AFP:', '').replace('�', '').replace(
        '- VNExplorer', '').replace('Global Tech News Daily', '').replace('AFP/TTXVN', '').replace('Reuters/TTXVN',
                                                                                                   '').replace(
        'Tin quốc tế', '').replace('Xem tiện ích cảnh báo giá CK', '')  # .replace('Reuters:','')

    if not any([w in full_text_clean[:10] for w in ["ASEAN, COVID"]]):
        full_text_clean = re.sub('^[A-Z ]{2,10}(,.{6,20})?(—|–|－|-)', '',
                                 full_text_clean)  # NEW YORK, Feb 27 — .... /BRUSSELS—...

    full_text_clean = re.sub('\(ảnh:.*?\)', '.', full_text_clean)
    full_text_clean = re.sub("(\| )?(\(.{1,7}\)( )?)+$", "", full_text_clean)

    full_text_clean = re.sub('\d{2} [\w]{3,4}, \d{4}. \d{2}.\d{2} (AM|PM) IST', '',
                             full_text_clean)  # 02 Mar, 2022, 10.01 AM IST

    full_text_clean = full_text_clean.replace(
        'Suzuka. config. supports_premium_subscription && window. localStorage. getItem ( "premiumSubscription ")) ) {var e = document. createElement ( "script "); e. setAttribute ( "class ", "titan-conditional "); e. setAttribute ( "data-ad-id ", "adspot-300x250-pos1 "); document. body. appendChild (e);}',
        '')

    full_text_clean = re.sub('\d{2}\/\d{2}\/\d{4} \d{2}:\d{2} GMT(\+|-)\d{1,2}', "", full_text_clean)


    full_text_clean = re.sub('[A-Z].{5,10} , \d{2}:\d{2} (GMT(\+|-)\d{1,2})?', "", full_text_clean)



    full_text_clean = re.sub('^\d{1,10} minutes ago', '', full_text_clean)
    full_text_clean = re.sub('^\d{1,10} hours ago', '', full_text_clean)
    full_text_clean = re.sub('^\d{1,10} days ago', '', full_text_clean)
    full_text_clean = re.sub('^\d{1,10} years ago', '', full_text_clean)
    full_text_clean = re.sub('^\d{1,10} months ago', '', full_text_clean)
    full_text_clean = re.sub('^\d{1,10} minute ago', '', full_text_clean)
    full_text_clean = re.sub('^\d{1,10} day ago', '', full_text_clean)
    full_text_clean = re.sub('^\d{1,10} year ago', '', full_text_clean)
    full_text_clean = re.sub('^\d{1,10} month ago', '', full_text_clean)
    full_text_clean = re.sub('^\d{1,10} hour ago', '', full_text_clean)
    full_text_clean = re.sub('^(a|an) minute ago', '', full_text_clean)
    full_text_clean = re.sub('^(a|an) hour ago', '', full_text_clean)
    full_text_clean = re.sub('^(a|an) day ago', '', full_text_clean)
    full_text_clean = re.sub('^(a|an) month ago', '', full_text_clean)
    full_text_clean = re.sub('^(a|an) year ago', '', full_text_clean)



    text = re.sub('\s+', ' ', full_text_clean)
    text = re.sub('Đọc chi tiết bài viết tại đây.*', '', text, flags=re.I)
    # text = re.sub('[(\d)(\:)(\|)(\/)(\s+)]+','',text) # 10:20 | 09/03/2022
    text = re.sub('(\d{1,2}:\d{2}( )*)\|( )*\d{1,2}(/|-)\d{2}(/|-)\d{4}', '', text)

    text = re.sub('^(\d)+[\,\.]\s+ ', '', text)  # 3, phát ngôn viên Bộ T
    text = re.sub(
        '((chủ nhật)|(thứ bảy)|(thử sáu)|(thứ năm)|(thứ tư)|(thứ ba)|(thứ hai))([(\d)(\:)(,)(\|\/)(\s+)]+)((VOV)|(VTV))$',
        '', text, flags=re.I)  # và Ukraine để giải quyết xung đột Chủ Nhật, 06:32, 20/03/2022 VOV.

    text = re.sub('^((\d)|(\:)|(\.)|(\|)|(\s+)|(in bài biết)|(in bài viết)|(\/))+', '', text,
                  flags=re.I)  # 10:20 | 09/03/2022 In bài biết. 10:20 | 09/03/2022 In bài biết Việc xuất khẩu tôm sang thị trường Nga có thể bị ảnh hưởng trước tình hình căng thẳng của Nga-Ukraine. Hiệp hội Chế biến và Xuất khẩu thuỷ sản V

    text = re.sub('video:RIA Novosti/Bộ Quốc phòng Nga', '', text, flags=re.I)
    text = re.sub('Báo.{5,20}$', '', text)
    text = re.sub('RIA Novosti/Bộ Quốc phòng Nga', '', text)
    text = re.sub('(chính trị|quân sự|đối ngoại|thời sự|khoa học|pháp luật) \d{1,2} giờ', '', text, flags=re.I)


    text = text.replace('|', '')
    full_text_clean = re.sub('^.*?(Link nguồn)', '', text, flags=re.I)  # (


    full_text_clean = re.sub('^VOV.', '', full_text_clean)
    full_text_clean = full_text_clean.replace("Baoquocte", "").replace('ชั่วโมงที่ผ่านมา.', '')

    full_text_clean = re.sub('(a|p)\. m\.', '', full_text_clean)
    full_text_clean = re.sub('This article was last updated at \d{1,2}:\d{2} (GMT/UTC)?', '', full_text_clean)

    full_text_clean = re.sub('(\* )mời quý độc giả .{5,}', "", full_text_clean, flags=re.I)
    full_text_clean = re.sub('Updated:.{3,10} \d{1,2}, \d{0,4} \d{1,2}:\d{2}:\d{2} (am|pm)?', '', full_text_clean)
    full_text_clean = full_text_clean.replace('related_posts_by_tax title= " "]', '').replace('Embed from Getty Images',
                                                                                              '').replace(
        'Written by Shubhajit Roy New Delhi', '').replace('By Associated Press', '').replace(
        'By:ENS Economic Bureau Mumbai', '').replace(
        'ha-vien-my-nancy-pelosi-bat-ngo-tham-ukraine Thứ 3 ngày 03/05/2022 Hotline:0272.3823225 Mail gửi tòa soạn Giới thiệu Liên hệ quảng cáo Kết nối NƯỚC CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM MUÔN NĂM! ĐẢNG CỘNG SẢN VIỆT NAM QUANG VINH MUÔN NĂM! Long An 30°C/27° 33°C Thế giới',
        '')

    full_text_clean = re.sub('^vn.( )?}', "", full_text_clean)
    full_text_clean = re.sub('^AFP.', "", full_text_clean)
    full_text_clean = re.sub('AA.', "", full_text_clean)
    full_text_clean = re.sub('^AFP.', "", full_text_clean)

    # full_text_clean = re.sub(".*?([A-Za-z]{3,4} \d{1,2}, \d{4} • \d{1,2} hours ago • \d{1,2} minute read) (• (\d{1,100} Comments|Join the conversation))","",full_text_clean)
    full_text_clean = re.sub('Il y a \d{1,5} heures( \. )?', "", full_text_clean)
    full_text_clean = full_text_clean.replace('Xem tiếp>>>', '').replace('\"', '"').replace('ชั่วโมงที่ผ่านมา .',
                                                                                            '').replace(
        '- Ảnh thời sự quốc tế - Chính trị-Quân sự - Thông tấn xã Việt Nam (TTXVN)', '').replace('TTXVN phát', '').replace(
        '| thefastnewz', '').replace('| Swotverge', '').replace('- VietBF', '').replace('THÔNG LUẬN - ', '').replace('vn. ',
                                                                                                                     "").replace(
        'minh hoạ: Getty', '')


    full_text_clean = re.sub('/detail/[^\s]*', '', full_text_clean)

    full_text_clean = re.sub("Use Next and Previous buttons to navigate (\d{1,2} )?(\d{1,2}of\d{1,2})?", '',
                             full_text_clean, flags=re.I)
    full_text_clean = re.sub(
        "(Advertisement)? \d{1,5} (Story continues below)? (This advertisement has not loaded yet, but your article continues below.)",
        "", full_text_clean, flags=re.I)
    full_text_clean = re.sub("Photo by .{1,15}/Article content W", "", full_text_clean, flags=re.I)
    full_text_clean = re.sub("\(photo by .*?\)", "", full_text_clean, flags=re.I)
    if "前" in full_text_clean:
        split_txt = full_text_clean.split("前")
        full_text_clean = ' '.join(split_txt[1:])
    if "heures" in full_text_clean:
        split_txt = full_text_clean.split("heures")
        full_text_clean = 'heures'.join(split_txt[1:])


    return full_text_clean


def preprocess(text:str):
    r = text
    try:
        # text = normalize_text(text)
        # text = text.replace('0:00 / 0:00 0:00 Nam miền Bắc Nam miền Bắc Nữ miền Bắc Nữ miền Nam Nam miền Nam','')
        text = text.replace('_',' ')
        # text = text.replace('‌', '')
        # replace segment
        # text.replace('_','')
        text = re.sub('\s+',' ',text)

        for punc in PUNCTUATION_SYM[1:]:
            text = text.lstrip(punc)
            text = re.sub(f'\s+\{punc}',f'{punc}',text)
            text = re.sub(f'\{punc}+',f'{punc}',text)
        text = text.strip()
    except Exception as ve:
        print(ve)
        print(r)
        print(text)
        return ''
    return text

# def sentence_breakdown(text:str):


def get_data():
    train_ratio = 0.7
    dev_ratio = 0.15
    test_ratio = 0.15
    train_origin_text_ls = []
    dev_origin_text_ls  = []
    test_origin_text_ls = []
    MIN_SENT_LENGTH = 10
    MAX_SENT_LENGTH = 126

    all_sentences_ls = []
    for category in category_list:
        print(category)
        with open(f'{folder_data_path}/{category}/content.json') as f:
            temp_ls = json.load(f)
            random.shuffle(temp_ls)

        temp_ls = [i['message'] for i in temp_ls if 'message' in i]
        temp_ls = list(map(preprocess,temp_ls))
        temp_ls = [i for i in temp_ls if i != '']
        random.shuffle(temp_ls)
        [all_sentences_ls.extend(sent_tokenize(i)) for i in temp_ls]
        del temp_ls

    data = []
    ind = 0
    while ind < len(all_sentences_ls):
        text = ''
        current_sentence = all_sentences_ls[ind]
        current_length = len(current_sentence.split())
        random_length = random.randint(MIN_SENT_LENGTH, MAX_SENT_LENGTH)

        while current_length <= random_length and ind < len(all_sentences_ls):
            text += current_sentence + ' '
            random_length -= current_length
            ind += 1
            if ind >= len(all_sentences_ls):
                break
            current_sentence = all_sentences_ls[ind]
            current_length = len(current_sentence.split())

        if random_length > 0 and ind < len(all_sentences_ls):
            current_sentence = all_sentences_ls[ind]
            text += ' '.join(current_sentence.split()[:random_length]) + ' '
            ind += 1

        text = text.strip()

        data.append(text)
    preprocessor = VietnameseNewsPreprocessor()
    with open('remove_word.txt') as f:
        remove_list_words = f.read().splitlines()

    data = preprocessor.data_cleaning(data,external_removal_list=remove_list_words)

    split_index1 = int(len(data)*train_ratio)
    split_index2 = split_index1 + int(len(data)*dev_ratio)

    print(split_index1,split_index2,len(data))
    train_origin_text_ls.extend(data[:split_index1])
    dev_origin_text_ls.extend(data[split_index1:split_index2])
    test_origin_text_ls.extend(data[split_index2:])

    print('train doc:',len(train_origin_text_ls))
    print('dev doc:',len(dev_origin_text_ls))
    print('test doc:',len(test_origin_text_ls))

    with open('data_new/train.txt','w') as f:
        f.write('\n'.join(train_origin_text_ls))
    with open('data_new/dev.txt','w') as f:
        f.write('\n'.join(dev_origin_text_ls))
    with open('data_new/test.txt','w') as f:
        f.write('\n'.join(test_origin_text_ls))

def random_show(path, number = 1000):
    with open(path) as f:
        ls = f.read().splitlines()[:number]
    with open('sample.txt', 'w') as f:
        f.write('\n'.join(ls))

if __name__ == '__main__':
    get_data()
    random_show('data_new/train.txt')

