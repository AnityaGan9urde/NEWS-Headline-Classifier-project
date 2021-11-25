import numpy as np
import pandas as pd
from flask import Flask, request, render_template, url_for
import joblib

from predict import predict_topic
from utils import vectorize

app = Flask(__name__)

lr_model = joblib.load('./models/lr_model.pkl')

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
    topic, prob_scores = predict_topic(text=text, model=lr_model)
    print(prob_scores)

    return render_template('index.html', prediction_text='The topic is most likely to be {}.'.format(topic))

if __name__ == '__main__':
    lr_model = joblib.load('./models/lr_model.pkl')
    app.run(debug=True)
