# VanHacks2020 Reviewsy (FINALISTS)
This application was created to be showcased at VanHacks 2020, a annual hackathon organized by TTT Studios around VSW dedicated to helping those helping our community. 
The team consists of three members: Jon Kim, Yen Ching Tan, and Gallant Tang.
We worked using python for both front and back end development on VSCode with live share due to social distancing policies as a result of covid.
# Introduction
The application has a number of functions. The main one is to build a topic model for restaurants based on their reviews. We collected >50,000 reviews on 1000 restaurants, cleaned the data using spaCy and nltk, and trained a LDA model using gensim to build out topics. Coherence testing was used to evaluate our model to ensure that the topics were accurate representations of any given restaurant.

# Goal
We built this to explore the applications of machine learning (natural language processing) and learned how to scrap websites. 
The purpose of building topic models for restaurants was to determine how a restaurants customers was profiling a certain restaurant. This is helpful as it will save restaurant owners a lot of time on secondary research. It is also a less biased way to view the reviews. Our vision for this product is to be able to generate a report for business owners to better understand how their customers view them, and what traits their restaurant possesses that their customers like/dislike the most.

## Data mining – Gallant
Gallant is responsible for creating the code used to mine data from Yelp's fusion API as well as other sources. We collected data on about 1000 restaurants in Vancouver and the 50000++ reviews left by their customers.

## Data preprocessing and analysis – Jon & Gallant
Gallant & Jon are responsible for cleaning, fitting, transforming, and training the LDA and NLP model using gensim LDA, NLTK, and spaCy.
We learned to create the `bag of words` through gensim for latent dirichlet analysis. 
Coherence testing and randomized search cv is used to optimize hyperparameters.

## UI/UX – Gallant & Yen
Yen & Gallant are responsible for developing the user interface. The application is created using Python Flask.
