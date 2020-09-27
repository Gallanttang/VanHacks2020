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
from gensim import corpora
import spacy
import re


#setting variables
#APIKEY = HEADERS['Authorization']
FILE_PATH = 'data/vancouver_reviews.csv'
#assuming that the data features will be: ['business_id', 'user_id', 'rating', 'comment', 'feedback', "time_uploaded"]

#retrieving data
reviews_df = pd.read_csv(FILE_PATH, index_col=0, delimiter='\t')
print(reviews_df.shape)
X_raw = reviews_df['comment']
X_raw.apply(lambda x: x.replace('<br>', '')).apply(lambda x: x.replace('<br', '')).apply(lambda x: x.replace('br>', ''))
y_raw = reviews_df['rating']

GOOD_RATING = 4

#cleaning the target data
y = y_raw>=GOOD_RATING

#print(y)

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


# to test what the tokens look like before applying LDA
import random
text_data = []
for line in X_raw:
    tokens = prepare_text_for_lda(line)
    if random.random() > -1:
        #print(tokens)
        text_data.append(tokens)

