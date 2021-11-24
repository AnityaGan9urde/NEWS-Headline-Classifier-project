import numpy as np

from preprocess import sent_to_words, lemmatization
from utils import vectorize

vectorizer = vectorize()

def predict_topic(text, nlp, model, vectorizer, kw):
    # Step 1: Clean with simple_preprocess
    text = list(sent_to_words(text))

    # Step 2: Lemmatize
    text = lemmatization(text, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Step 3: Vectorize transform
    text = vectorizer.transform(text)

    # Step 4: LDA Transform
    topic_probability_scores = model.transform(text)
    topic = kw.iloc[np.argmax(topic_probability_scores), :].values.tolist()
    return topic, topic_probability_scores
