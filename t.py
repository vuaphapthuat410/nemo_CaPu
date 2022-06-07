import string
import re
all_punct_marks = '!,.?:;'
# all_punct_marks = string.punctuation.replace("'", '')

print(all_punct_marks)
print('[' + all_punct_marks + ']')
# return re.sub('[' + all_punct_marks + ']', '', word)
word = 'Kết quả giám định pháp y về thương tích đối với anh N. là 19%. Hiện cơ quan công an (đang)'
print(re.sub('[' + all_punct_marks + ']', '', word))