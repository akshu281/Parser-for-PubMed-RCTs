import pymedtermino as pt
import json
from nltk.corpus import stopwords
import nltk
import string

res_list=[]
obj_list=[]
meth_list=[]
bg_list=[]
conc_list=[]
trans=str.maketrans('','', string.punctuation)
data_path="D:/Lecture Notes/Semester 2/Text Mining/CA - 1/codebase/text/"
train_file=data_path+'train.json'

with open(train_file, 'r') as f:
    lines_json= json.load(f)
# print(type(lines_json))
for d in lines_json:
    if d['label'] == "RESULTS":
        res_list.append(json.dumps(d['text']))
    elif d['label'] == "OBJECTIVE":
        obj_list.append(json.dumps(d['text']))
    elif d['label'] == "METHODS":
        meth_list.append(json.dumps(d['text']))
    elif d['label'] == "CONCLUSIONS":
        conc_list.append(json.dumps(d['text']))
    elif d['label'] == "BACKGROUND":
        bg_list.append(json.dumps(d['text']))

stopwords = stopwords.words('english')

tokens = []
    # print("inputlist", len(inputlist))
    
for d in meth_list[:10]:
    data_tokens = nltk.wordpunct_tokenize(d)
    # unique = set(data_tokens)
    # print(len(data_tokens))
    single=[w for w in data_tokens if len(w) == 1]
    # print(len(single))
    # data_tokens_f=[tok for tok in data_tokens if tok not in string.punctuation]
    data_tokens_f=[tok for tok in data_tokens if tok not in string.punctuation and tok not in single]
    # print("Tokens after punc and single char removal")
    # print(len(data_tokens))
    # filtered = [word.translate(trans) for word in data_tokens_f if word.lower() not in stopwords and not word.isdigit()]
    filtered = [word for word in data_tokens_f if word.lower() not in stopwords and not word.isdigit()]
    tokens = tokens + filtered

# dicto=pt.get_concept("icd10:I10")
# print(dicto)
# dicto=pymedtermino.cdf
# print(dicto)
# load()