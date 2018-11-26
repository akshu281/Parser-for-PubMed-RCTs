from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora

data_path="D:/Lecture Notes/Semester 2/Text Mining/CA - 1/codebase/text/"
train_file=open(data_path+'train.txt')
lines=train_file.readlines()
print(len(lines))

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(line):
    stop_free = " ".join([i for i in line.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(line).split() for line in lines]
print("Pre-Processed Document")
print(len(doc_clean))

# Term dictionary - unique term with an index 
dictionary = corpora.Dictionary(doc_clean)
dictionary.save('dictionary.gensim')
# List of documents into DTM
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
print("Doc Term Matrix")
print(len(doc_term_matrix))

# print("Five elements in list")
# for i in doc_term_matrix:
#     print(str(i))

# Creating the object for LDA model using gensim library
lda_model = gensim.models.ldamodel.LdaModel
lda_model.save('lda_model.gensim')
# Running and Trainign LDA model on the document term matrix.
ldamodel = lda_model(doc_term_matrix, num_topics=1, id2word = dictionary, passes=50)
print("LDA Model")
print(ldamodel.print_topics(num_words=10))
