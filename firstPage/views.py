from django.shortcuts import render
from django.http import HttpResponse

import tensorflow as tf
from transformers import BertTokenizer
import numpy as np
import joblib, pickle

bert_model = tf.keras.models.load_model('./models/news_model.h5')
lr_model = joblib.load('./models/lr_model.pkl')

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
vectorizer = joblib.load('./models/vectorizer.pkl')

no_to_topic = {'CRIME': 0, 'ENTERTAINMENT': 1, 'WORLD NEWS': 2, 'IMPACT': 3, 'POLITICS': 4,
                            'WEIRD NEWS': 5, 'BLACK VOICES': 6, 'WOMEN': 7, 'COMEDY': 8, 'QUEER VOICES':9,
                            'SPORTS':10, 'BUSINESS': 11, 'TRAVEL': 12, 'MEDIA': 13, 'TECH': 14, 'RELIGION': 14,
                            'SCIENCE': 15, 'LATINO VOICES': 16, 'EDUCATION': 17, 'COLLEGE': 18, 'PARENTS': 19,
                            'ARTS & CULTURE': 20, 'STYLE': 21, 'GREEN': 22, 'TASTE': 23, 'HEALTHY LIVING': 24,
                            'THE WORLDPOST': 25, 'GOOD NEWS': 26, 'WORLDPOST': 27, 'FIFTY': 28, 'ARTS': 29,
                            'WELLNESS': 30, 'PARENTING': 31, 'HOME & LIVING': 32, 'STYLE & BEAUTY': 33,
                            'DIVORCE': 34, 'WEDDINGS': 35, 'FOOD & DRINK': 36, 'MONEY': 37, 'ENVIRONMENT': 38,
                            'CULTURE & ARTS': 39}

# Create your views here.

def index(request):
    return render(request, 'index.html')

def predict(request):
    if request.method == 'POST':
        headline = request.POST.get('headline') # input by user
        model = request.POST.get('model') # which model to use?
        quant = request.POST.get('quant')
        print('headline:', headline, 'model:', model, 'quant:', quant)
    if model=='Bert':
        tokens = tokenizer.encode_plus(headline, 
                                        max_length=70, 
                                        truncation=True, 
                                        padding='max_length', 
                                        add_special_tokens=True, 
                                        return_token_type_id=False, 
                                        return_tensors='tf')

        test = {'input_ids': tf.cast(tokens['input_ids'], tf.float64), 
                'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)}

        probs = bert_model.predict(test) # Probability scores by the model for all the topics

        probs = np.array(probs[0]) # converting into a numpy array
        if quant=='5':    
            pred = probs.argsort()[-5:][::-1] # sorting and taking indexes for top 5 highest values
        elif quant=='10':
            pred = probs.argsort()[-10:][::-1]
        elif quant=='20':
            pred = probs.argsort()[-20:][::-1]
        else:
            pred = probs.argsort()[-5:][::-1]

        pos = []
        for i in pred:
            val = list(no_to_topic.values()).index(i) # finding indexes for values in a values list of topics dict
            pos.append(list(no_to_topic.keys())[val]) # finding keys from a keys list of topics dict by using val

        if quant=='5':    
            vals = np.sort(probs)[-5:][::-1].tolist() # sorting and taking indexes for top 5 highest values
        elif quant=='10':
            vals = np.sort(probs)[-10:][::-1].tolist()
        elif quant=='20':
            vals = np.sort(probs)[-20:][::-1].tolist()
        else:
            vals = np.sort(probs)[-5:][::-1].tolist()

        vals = [(round(x, 6))*100 for x in vals]
        print(vals)
        for i in range(5):
            print(f'Prediction: {pos[i]} with a prob of {vals[i]}%')

    elif model=='LR':
        tokens = vectorizer.transform([headline])
        probs = lr_model.predict_proba(tokens)
        probs = np.array(probs[0])
        
        if quant=='5':    
            pred = probs.argsort()[-5:][::-1] # sorting and taking indexes for top 5 highest values
        elif quant=='10':
            pred = probs.argsort()[-10:][::-1]
        elif quant=='20':
            pred = probs.argsort()[-20:][::-1]
        else:
            pred = probs.argsort()[-5:][::-1]

        pos = []
        for i in pred:
            val = list(no_to_topic.values()).index(i)
            pos.append(list(no_to_topic.keys())[val])

        if quant=='5':    
            vals = np.sort(probs)[-5:][::-1].tolist() # sorting and taking indexes for top 5 highest values
        elif quant=='10':
            vals = np.sort(probs)[-10:][::-1].tolist()
        elif quant=='20':
            vals = np.sort(probs)[-20:][::-1].tolist()
        else:
            vals = np.sort(probs)[-5:][::-1].tolist() 

        vals = [(round(x, 6))*100 for x in vals]
        for i in range(5):
            print(f'Prediction: {pos[i]} with a prob of {vals[i]}%')

    op_dict = dict(zip(pos, vals))
    context = {'op_dict': op_dict}
    print('op_dict:', op_dict)

    return render(request, 'index.html', context)

