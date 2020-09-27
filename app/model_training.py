
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
import review_preprocessing as rp

#LDA with Gensim
#from https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21

dictionary = corpora.Dictionary(rp.text_data)
corpus = [dictionary.doc2bow(text) for text in rp.text_data]

import pickle
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

import gensim
NUM_TOPICS = 5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)

def LDA_analysis(new_doc):
    new_doc = rp.prepare_text_for_lda(new_doc)
    new_doc_bow = dictionary.doc2bow(new_doc)
    print(new_doc_bow)
    print(ldamodel.get_document_topics(new_doc_bow))

LDA_analysis("STRICKLAND: Her mother wrote the stances.")

#so there's a few things
#i think we should run this on all the restaurants  (maybe like 15 categories)
# they should be marked as 'in that category' if they're above 40% or 50% or something
# there's two things we could do, we could:
# # take the business features and tell them which category they should pursue
# # or we could parse through their reviews, identify their category and share the best practices of the top performers in that category
# i think the second one is more interesting
# that means we'd have to be able find the top performers in each category