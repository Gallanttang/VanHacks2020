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
import lda_model_training as mp
import scipy.stats
import matplotlib.pyplot as plt

FILE_PATH_REVIEWS = 'data/vancouver_reviews.csv'
FILE_PATH_RESTAURANTS = 'data/Restaurant_links.csv'

reviews_df = pd.read_csv(FILE_PATH_REVIEWS, index_col=0, delimiter='\t')
restaurants_df = pd.read_csv(FILE_PATH_RESTAURANTS, index_col=0)


print(reviews_df.shape)
print(restaurants_df.shape)

X_raw = reviews_df['comment']
X_raw.apply(lambda x: x.replace('<br>', ''))
y_raw = reviews_df['rating']

with open('data/Restaurant_categorization.csv', mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter='\t')
    csv_writer.writerow(['id', 'reviews', 'target', 'primary', 'secondary', 'tertiary'])
    #jon finished here
    for i in range(20):
        try:
            PARAMETERS['offset'] = i * 50
            r = requests.get(url = ENDPOINT, params = PARAMETERS, headers=HEADERS)
            json_loaded = json.loads(r.text)
            print(len(json_loaded['businesses']))
            if json_loaded['businesses'] is None or len(json_loaded['businesses']) < 1:
                break
            for business in json_loaded['businesses']:
                price = ""
                if 'price' in business.keys():
                    price = business['price']
                csv_writer.writerow([business['id'], business['name'],
                                    business['url'], business['location']['city'],
                                    business['location']['zip_code'], 
                                    business['rating'],
                                    business['categories'],
                                    price
                                    ])
        except:
            print("failed, moving on.")
            break