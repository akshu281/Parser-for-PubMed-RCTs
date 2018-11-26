
# coding: utf-8

# In[23]:


import sys
import os
import re

import nltk
from nltk.corpus import stopwords

import gensim.models
from bs4 import BeautifulSoup

import warnings
STOP_WORDS = set(stopwords.words("english"))


# In[24]:


def get_file_contents(fname):
    """
    Returns contents from input file
    """
    with open(fname) as f:
        rows = f.readlines()
    return rows


# In[25]:


def get_parsed_data(raw_data):
    """Function parses raw data and return the parsed data"""
    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    data = []
    for raw_d in raw_data:
        data += review_to_sentences(raw_d, tokenizer)
    return data


# In[26]:


def get_parsed_data(raw_data):
    """Function parses raw data and return the parsed data"""
    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    data = []
    for raw_d in raw_data:
        data += review_to_sentences(raw_d, tokenizer)
    return data


# In[27]:


def review_to_wordlist(review, remove_stopwords=False):
    """
    Function removes HTML, Non-letters and optionally Stopwords and converts
    words to lower case and splits them into a list of words. The list of words
    is returned
    """

    # 1. Remove HTML
    #review_text = remove_html(review)
    review_text = review
    
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)

    # 3. Convert words to lower case and split them
    words = str(review_text.lower()).split()

    # 4. Remove stopwords
    if remove_stopwords:
        words = [w for w in words if w not in STOP_WORDS]
    return words


# In[28]:


def review_to_sentences(review, tokenizer, remove_stopwords=True):
    """
    Define a function to split a review into parsed sentences, where
    each sentence is a list of words. Fucntion returns the parsed data
    """
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())

    # 2. Loop over each sentence
    sentences = []
    for sentence in raw_sentences:
        if len(sentence) > 0:
            words = review_to_wordlist(sentence, remove_stopwords)
        sentences.append(words)

    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


# In[29]:


data=[]
for f in ('train.txt','test.txt','dev.txt'):
        file_path =  f
        print(file_path)
        raw_data = get_file_contents(file_path)
        parsed_data = get_parsed_data(raw_data)
        data += parsed_data


# In[31]:


data


# In[30]:


len(data)


# In[32]:


def train_model(train_data):
    """Function trains and creates Word2vec Model using parsed
    data and returns trained model"""
    model = gensim.models.Word2Vec(train_data, min_count=2)
    return model



# In[33]:



def get_w2v_model(data):
    """Function creates trained model and saved model persistently to disk"""
    model_name = "word2vec_model"
    trained_model = train_model(data)
    trained_model.save(model_name)
    print("Saved %s model successfully" % model_name)


# In[34]:


w2v_model = get_w2v_model(data)

