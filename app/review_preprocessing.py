import numpy as np
import pandas as pd
import nltk
import gensim
from gensim import corpora
import spacy
import re
FILE_PATH = 'data/vancouver_reviews.csv'



class CleanedData:
    def __init__(self):
        # setting variables
        # APIKEY = HEADERS['Authorization']
        # assuming that the data features will be: ['business_id', 'user_id', 'rating', 'comment', 'feedback', "time_uploaded"]

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
        self.parser = English()
        nltk.download('wordnet')
        from nltk.corpus import wordnet as wn
        self.wn = wn
        from nltk.stem.wordnet import WordNetLemmatizer


        nltk.download('stopwords')
        self.en_stop = set(nltk.corpus.stopwords.words('english'))

        import random
        text_data = []
        for line in X_raw:
            tokens = self.prepare_text_for_lda(line)
            if random.random() > -1:
                #print(tokens)
                text_data.append(tokens)

        dictionary = corpora.Dictionary(text_data)

        corpus = [dictionary.doc2bow(text) for text in text_data]

        import pickle
        pickle.dump(corpus, open('corpus.pkl', 'wb'))
        dictionary.save('dictionary.gensim')

        import gensim
        NUM_TOPICS = 13
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
        ldamodel.save('model5.gensim')
        topics = ldamodel.print_topics(num_words=15)
        for topic in topics:
            print(topic)


    def prepare_text_for_lda(self, text):
        tokens = self.tokenize(text)
        tokens = [token for token in tokens if len(token) > 4]
        tokens = [token for token in tokens if token not in self.en_stop]
        tokens = [self.get_lemma(token) for token in tokens]
        return tokens

    def tokenize(self, text):
        lda_tokens = []
        tokens = self.parser(text)
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

    def get_lemma2(self, word):
        return WordNetLemmatizer().lemmatize(word)

    def get_lemma(self, word):
        lemma = self.wn.morphy(word)
        if lemma is None:
            return word
        else:
            return lemma