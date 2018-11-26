import os
import csv
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
import pandas as pd
import itertools
import json
import csv
import wordcloud
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

diseaselist=[]
disease=[]

def load_data():
  with open('outfile.csv', 'r') as f:
    reader = csv.reader(f)
    disease = list(reader)
  temp = list(itertools.chain(*disease))
  train_file=open('train.txt')
  lines=train_file.readlines()
  # print(len(lines))
  for i in temp:
    diseaselist.append(i)
  freq(lines)

hit_list=[]

def freq(lines):
  # print(len(diseaselist))
  for l in lines: 
    for d in diseaselist:
      # print(d)
      if d in l:
        hit_list.append(d)
      else:
        continue

load_data()

# print("Total Diseases Hit")
# print(len(hit_list))

dis=[]

for i in hit_list:
  # print(i)
  dis.append({'disease': i,'freq':hit_list.count(i)})

with open('disease_dict.json', 'w') as f:
    json.dump(dis, f)

value= pd.read_json('disease_dict.json')
temp = value.drop_duplicates('disease')
# print(temp)
temp.to_csv('diseasescount.csv',index=False)

data = csv.reader(open('diseasescount.csv', 'r',newline='\n'))
d = {}
next(data)
for k,v in data:
  # print(v)
  d[k] = float(v)
wordcloud = WordCloud().generate_from_frequencies(d)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()