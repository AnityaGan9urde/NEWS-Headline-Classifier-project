import numpy as np
import pickle

def vectorize():
    with open('./models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer
