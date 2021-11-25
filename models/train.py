import pandas as pd
import numpy as np
import pickle, joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('train.csv')

df['category_id'] = df['Category'].factorize()[0]

vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = vectorizer.fit_transform(df.Text).toarray()
labels = df.category_id

model = LogisticRegression(random_state=0)

#Train Algorithm
lr_model = model.fit(features, labels)

joblib.dump(lr_model, 'lr_model.pkl')

with open('vectorizer.pkl', 'wb') as fout:
    pickle.dump((vectorizer), fout)

