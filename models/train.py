import numpy as np
import pandas as pd
import re, nltk, spacy, gensim
import joblib, pickle

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from utils import show_topics
from preprocess import preprocess, sent_to_words, lemmatization

seed = 100

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

df = pd.read_csv('./data/train.csv')

data = preprocess(df)

data_words = list(sent_to_words(data))

data_lemmatized = lemmatization(data_words, nlp=nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

vectorizer = CountVectorizer(analyzer='word',
                             min_df=10,                        # minimum read occurences of a word
                             stop_words='english',             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             # max_features=50000,             # max number of uniq words
                            )

data_vectorized = vectorizer.fit_transform(data_lemmatized)

lda_model = LatentDirichletAllocation(n_components=5,               # Number of topics
                                      max_iter=10,               # Max learning iterations
                                      learning_method='online',
                                      random_state=seed,          # Random state
                                      batch_size=128,            # n docs in each learning iter
                                      evaluate_every=-1,       # compute perplexity every n iters, default: Don't
                                      n_jobs=-1,               # Use all available CPUs
                                      learning_decay=0.9
                                     )

lda_output = lda_model.fit_transform(data_vectorized)


topic_keywords = show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=15)

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords.to_csv('df_topic_keywords.csv')

"""
Saving the model and the vectorizer
"""

joblib.dump(lda_model, 'lda_model.jl')

with open('vectorizer.pkl', 'wb') as fout:
    pickle.dump((vectorizer), fout)
