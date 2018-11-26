import nltk
from nltk.collocations import *
import string
from nltk.probability import FreqDist
import json
from textblob import TextBlob
from nltk import word_tokenize, pos_tag
import itertools
from nltk.corpus import stopwords

res_list=[]
obj_list=[]
meth_list=[]
bg_list=[]
conc_list=[]

trans=str.maketrans('','', string.punctuation)
data_path="D:/Lecture Notes/Semester 2/Text Mining/CA - 1/codebase/text/"
train_file=data_path+'train.json'

def pos(sent, flag):
    # print(type(sent))
    words = []
    pos_family = {
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adv' : ['RB','RBR','RBS','WRB']
    }
    cnt = 0
    # print(len(sent))
    # for content in sent:
    # print(content)
    try:
        wiki = TextBlob(sent)
        # print(wiki)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                words.append(list(tup)[0])
                cnt += 1
    except:
        pass
    # print(words)
    return words

words_f=[]
def load():
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
        
    print("List length of 5 classes")
    print(len(res_list))
    print(len(obj_list))
    print(len(meth_list))
    print(len(conc_list))
    print(len(bg_list))

    print("POS Tagging of 5 classes")
    tot_list=res_list+obj_list+meth_list+conc_list+bg_list
    print(len(tot_list))
    words_final=[]
    for i_ in tot_list[:100]:
        # print(type(i_))
        words_final= words_final + pos(i_,'noun')
        words_final= words_final + pos(i_,'verb')
        words_final= words_final + pos(i_,'pron')
    
    words_f=words_final
    print("More Stopwords")
    print(len(words_f))

    print("Bigrams of 5 classes")
    tokens_res=bigrams(res_list,name="RESULTS")
    tokens_obj=bigrams(obj_list,name="OBJECTIVE")
    tokens_meth=bigrams(meth_list,name="METHODS")
    tokens_conc=bigrams(conc_list,name="CONCLUSIONS")
    tokens_bg=bigrams(bg_list,name="BACKGROUND")

    fd = nltk.FreqDist(tokens_res)
    fd.most_common(50)
    fd.plot(50)
    
    fd = nltk.FreqDist(tokens_obj)
    fd.most_common(50)
    fd.plot(50)
    
    fd = nltk.FreqDist(tokens_meth)
    fd.most_common(50)
    fd.plot(50)
    
    fd = nltk.FreqDist(tokens_conc)
    fd.most_common(50)
    fd.plot(50)
    
    fd = nltk.FreqDist(tokens_bg)
    fd.most_common(50)
    fd.plot(50)
    
stopwords = stopwords.words('english') + ['.\\','Patient','Patients','Group','Groups','group','groups','patient','patients','vs'] + words_f

def bigrams(inputlist,name):
    ''' Read input data '''
    tokens = []
    # print("inputlist", len(inputlist))
    
    for d in inputlist:
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

    json.dump(tokens,'token.json')
    print('\nTotal tokens loaded: %d' % (len(tokens)))
    # print('Calculating Collocations')

    ''' Extract bigrams within a window of 5'''
    bigram_measure = nltk.collocations.BigramAssocMeasures()
    # finder_bi = BigramCollocationFinder.from_words(tokens, 5))
    finder_bi = BigramCollocationFinder.from_words(tokens)
    finder_bi.apply_freq_filter(5)
    # fdist = FreqDist(finder_bi)
    # print("Frequency Dist - Bigrams")
    # for k,v in finder_bi.ngram_fd.items():
    #   print(k,v)

    ''' Return top n bi-grams'''
    print("Printing Top 50 bi-grams Collocations")
    print("Bi-grams for "+ name +":\n", sorted(finder_bi.nbest(bigram_measure.phi_sq, 50)))
    # print(finder_bi.nbest(bigram_measure.raw_freq, 50))
    # print(finder_bi.nbest(bigram_measure.likelihood_ratio, 50))
    # print(finder_bi.nbest(bigram_measure.chi_sq, 50))

    ''' Return top n tri-grams'''
    # trigram_measure = nltk.collocations.TrigramAssocMeasures()
    # finder_tri = TrigramCollocationFinder.from_words(tokens,5)
    # finder_tri.apply_freq_filter(5)
    # print('Printing Top 20 tri-grams Collocations')
    # print(finder_tri.nbest(trigram_measure.likelihood_ratio, 50))
    # print(finder_tri.nbest(trigram_measure.chi_sq, 50))
    #print(finder_tri.nbest(trigram_measure.phi_sq, 50))
    # print(finder_tri.nbest(trigram_measure.raw_freq, 50))
    return tokens

load()