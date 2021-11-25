import numpy as np
import pandas as pd
from flask import Flask, request, render_template, url_for
import joblib, spacy
from sklearn.feature_extraction.text import CountVectorizer

from predict import predict_topic
from utils import vectorize

app = Flask(__name__)

lda_model = joblib.load('./models/lda_model.jl')

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

topic_dict = {'0': 'politics', '1': 'technology', '2': 'sports', '3': 'entertainment', '4': 'business'}

vectorizer = vectorize()

df_topic_keywords = pd.read_csv('df_topic_keywords.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    d = None
    if request.method == 'POST':
        print('POST Received.')
        d = request.form.to_dict()
    else:
        print('GET Received.')
        d = request.args().to_dict()

    text = [str(d['article'])]
    topic, prob_scores = predict_topic(text=text, nlp=nlp, model=lda_model, vectorizer=vectorizer, kw=df_topic_keywords)

    prob_scores = prob_scores.tolist()
    prob_scores = prob_scores[0]

    preds = sorted(prob_scores, reverse=True)

    first_topic_prob = preds[0]
    second_topic_prob = preds[1]
    i = prob_scores.index(first_topic_prob)
    j = prob_scores.index(second_topic_prob)
    first_topic = topic_dict[str(i)]
    second_topic = topic_dict[str(j)]

    if second_topic_prob>0.38:
        return render_template('index.html', prediction_text='The topic is most likely to be {} and second most likely to be {}.'.format(first_topic, second_topic))
    else:
        return render_template('index.html', prediction_text='The topic is most likely to be {}.'.format(first_topic))

if __name__ == '__main__':
    lda_model = joblib.load('./models/lda_model.jl')
    app.run(debug=True)
