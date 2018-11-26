# -*- coding: utf-8 -*-

import nltk
from nltk import word_tokenize,pos_tag,ne_chunk
from nltk.tag.stanford import StanfordPOSTagger, StanfordNERTagger
from stanfordcorenlp import StanfordCoreNLP
import json
import nltk
import logging

# nltk.download('propbank')
# from nltk.corpus import propbank



data_path="D:/Lecture Notes/Semester 2/Text Mining/CA - 1/codebase/text/"
nlp = StanfordCoreNLP('stanford-corenlp-full-2018-02-27',quiet=True)
# instances=propbank.instances()

def POStagging(sentences):
    for sentence in sentences:
        # sentence = '''To investigate the efficacy of 6 weeks of daily low-dose oral prednisolone in improving pain , mobility , and systemic low-grade inflammation in the short term and whether the effect would be sustained at 12 weeks in older adults with moderate to severe knee osteoarthritis ( OA )
        # '''
        # print("\n",sentence)
        # print("\nTokenized Input")
        # print(nlp.word_tokenize(sentence))
        # print("\nPOS Tagging - StandfordCoreNLP")
        # print(nlp.pos_tag(sentence))
        # print("\nPOS Tagging - NLTK")
        # sent_pos = pos_tag(word_tokenize(sentence))
        # print(sent_pos)
        NERtagging(sentence)
        #NERTagging_try(sentence)
              
def NERtagging(pos):
    print("\n",pos)
    # for pos in pos_list:
    print("\nNamed Entities")
    # print(nlp.ner(pos))
    sent_chunk = nlp.ner(pos)
    print(sent_chunk)

# def NERTagging_try(pos):
#     ner_model_path = 'D:/Lecture Notes/Semester 2/Text Mining/Day 4/stanford-ner-2018-02-27/classifiers/english.all.3class.distsim.crf.ser.gz'
#     ner_jar_path = 'D:/Lecture Notes/Semester 2/Text Mining/Day 4/stanford-ner-2018-02-27/stanford-ner.jar'

#     st_ner = StanfordNERTagger(ner_model_path, ner_jar_path)
#     sent_ne = st_ner.tag(word_tokenize(pos))
#     print("NER Tagging Try")
#     print(sent_ne)

def load_data():
    train_file=open(data_path+'train.txt')
    lines=train_file.readlines()
    print(len(lines))
    POStagging(lines[0:5])


load_data()