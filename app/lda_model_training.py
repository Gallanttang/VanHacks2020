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
import scipy.stats
import matplotlib.pyplot as plt


#LDA with Gensim
#from https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21

class LDA_ANALYST:
    print(1)
    cd = rp.CleanedData()
    def LDA_analysis(self, new_doc):
        new_doc = self.cd.prepare_text_for_lda(new_doc)
        new_doc_bow = dictionary.doc2bow(new_doc)
        print(new_doc_bow)
        print(ldamodel.get_document_topics(new_doc_bow))
        return [new_doc_bow, ldamodel.get_document_topics(new_doc_bow)]

lda = LDA_ANALYST()

lda.LDA_analysis("Medina Cafe is a cute/cozy spot for brunch. We visited during a snowstorm and the place was still packed, which blew our minds. Since there were so many rave reviews for this place, we figured we'd see what all the hype was about.<br><br>I had the Le Complet (2 Sunny Eggs, Flatiron Steak, Toasted Focaccia with Caramelized Onion and Roasted Red Pepper Chèvre, Romesco, Organic Greens) and hubby had the Fricassée (2 Sunny Eggs, Braised Short Ribs, Roasted Potatoes, Caramelized Onions, Smoked Cheddar, Arugula, Grilled Focaccia). <br><br>Both arrived beautifully plated and were delicious. We ended up sharing a waffle because I guess you can't come here without trying one, but were totally disappointed that it arrived ice cold. We ended up sending it back and they warmed it up for us, but it would've been nice if it had been freshly made in the first place. <br><br>Cute spot, good food, hectic atmosphere. Would visit again to try another waffle.")
lda.LDA_analysis("Sushi")
lda.LDA_analysis("Chocolate")
#so there's a few things
#i think we should run this on all the restaurants  (maybe like 15 categories)
# they should be marked as 'in that category' if they're above 40% or 50% or something
# there's two things we could do, we could:
# # take the business features and tell them which category they should pursue
# # or we could parse through their reviews, identify their category and share the best practices of the top performers in that category
# i think the second one is more interesting
# that means we'd have to be able find the top performers in each category