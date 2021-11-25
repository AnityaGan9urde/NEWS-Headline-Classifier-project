import numpy as np
from utils import vectorize

vectorizer = vectorize()

id_to_category = {0: 'business', 1: 'technology', 2: 'politics', 3: 'sports', 4: 'entertainment'}

def predict_topic(text, model):
    text = vectorizer.transform(text)
    topic_probability_scores = model.predict_proba(text)
    prediction = model.predict(text)
    topic = id_to_category[prediction[0]]
    return topic, topic_probability_scores
