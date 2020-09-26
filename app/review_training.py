import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import nltk
import gensim
import spacy
import re


#setting variables
#APIKEY = HEADERS['Authorization']
FILE_PATH = 'data/vancouver_reviews.csv'
#assuming that the data features will be: ['businessID', 'userID', 'rating', 'review', 'time']

#retrieving data
# reviews_df = pd.read_csv(FILE_PATH, index_col=0)
# print(reviews_df.shape)
# X_raw = reviews_df['review']
# y_raw = reviews_df['ratings']

GOOD_RATING = 4
y_raw = ['3.4','4', 'hella 3', '2 a3glsk2']


X_raw = ["STRICKLAND: Good morning.",

"Marsha is on her way. She called from the car phone I think. It sounded like the car phone, to let us know that she would be delayed.",

"I would like to welcome two people who haven't been with us before.",

"Suzanne Clewell, we're delighted to have you with us today. Suzanne, would you tell us a little bit about what you do?",

"CLEWELL: Yes. I'm the Coordinator for Reading Language Arts with the Montgomery County Public Schools which is the suburban district surrounding Washington. We have 173 schools and 25 elementary schools.",

"It's great to be here.",

"STRICKLAND: And I'll skip over to another member of the committee, but for her, this is her first meeting, too, Judith Langer. I think we all know her work, if we didn't know her.",

"Judith.",

"LANGER: Hello. I'm delighted to be here.",

"I have carefully read and heard about all of the things that the group has discussed up until now.",

"I'm a Professor of Education at the University of Albany, the State University of New York. And I'm also the Director of the National Research Center on English Learning and Achievement.",

"STRICKLAND: Her mother wrote the stances."]

#cleaning the target data
y = []
for s in y_raw:
    l = []
    for t in s.split():
        try:
            l.append(float(t))
        except ValueError:
            pass
    y.append(l[0]>=GOOD_RATING)
print(y)
 
# preprocessing the corpus
from spacy.lang.en import English

parser = English()

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

nltk.download('wordnet')
from nltk.corpus import wordnet as wn

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

import random
text_data = []
for line in X_raw:
    tokens = prepare_text_for_lda(line)
    
    if random.random() > .99:
        print(tokens)
        text_data.append(tokens)

#from https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21


# X = []
# for i in X_raw:
#     X.append(gensim.utils.simple_preprocess(i, deacc=True, min_len=3))
# bigram = gensim.models.Phrases(X_raw)
# stops = set(stopwords.words('english'))

# def process_texts(texts):
#     texts = [[word for word in line if word not in stops] for line in texts]
#     texts = [bigram[line] for line in texts]
#     texts = [[word.decode("utf-8").split('/')[0] for word in lemmatize(' '.join(line), allowed_tags=re.compile('(NN)'), min_length=5)] for line in texts]
#     return texts

# train_texts = process_texts(X_raw)

# dictionary = Dictionary(train_texts)
# corpus = [dictionary.doc2bow(text) for text in train_texts]
# print(Dictionary)
# print(corpus)
