import numpy as np
import pickle

def show_topics(vectorizer, lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

def vectorize():
    with open('./models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer
