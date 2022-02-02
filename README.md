# NEWS Headline Classifier 
- A Django app to determine the topic for a News article headline / title using the BERT model.
- Can detect around 40 topics for a title.  
---
### :warning: Note that the trained Bert model is not included in this repository due to GitHub file size limits.
---
## App preview:<br>
<img src="https://github.com/AnityaGan9urde/NEWS-Headline-Classifier-project/blob/main-main/app_preview.gif"></img>
## Dataset:
- The data used for training was taken from Kaggle.
- Dataset link: https://www.kaggle.com/rmisra/news-category-dataset.
- The dataset consists of around 200,000 rows of News Article related information.
## Preprocessing:
- I kept only two columns from the dataset: `df.headline` and `df.category`. 
- I imported a pretrained tokenizer from the `BertTokenizer` module called `'bert-base-cased'` and passed the df.headline into it to generate tokens.
- The tokens consisted of `input_ids` and `attention_masks` which I then assigned to null matrices called Xids and Xmask.
- For encoding of labels, I created a null matrix of the shape=(no. of samples, no. of categories), and then placed `1` for every row at a column number corresponding to the topic.
- The dataset was then shuffled and batches were created to pass it to the model.
## Model:
- I selected the BERT model for training this dataset on as it is currently the best model for NLP related tasks.
- BERT, which stands for Bidirectional Encoder Representation from Transformers, is a transformer-based deep learning technique developed for all kinds of natural language processing tasks.
- It is highly efficient, easy to setup and train your data on.
- I downloaded the 'bert-base-cased' model and modified it as such:
```python
def create_model():
    bert = TFAutoModel.from_pretrained('bert-base-cased')
    input_ids = tf.keras.layers.Input(shape=(seq_len,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(seq_len,), name='attention_mask', dtype='int32')

    embeddings = bert.bert(input_ids, attention_mask=mask)[1]

    x = tf.keras.layers.Dense(1024, activation='relu')(embeddings)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    y = tf.keras.layers.Dense(arr.max()+1, activation='softmax', name='outputs')(x)

    model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)
    
    return model

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-6)
loss = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[acc])
```

## Training:
- I trained the model for 8 epochs and achieved a training accuracy of 92% and a validation accuracy of 90%.
- The training was carried on for 9 hours on an Nvidia 1050 Ti GPU.
## Django App:
- I wrapped the model in a django app and created an interface where users can interact with the model.
- Users have an option to select from BERT model and a Logistic Regression base model. They can also choose the top 'N' predictions they want for a headline.
## Deployment:
- The app can be Dockerize and deployed on an AWS instance or a GCP instance.
- Each deploment will require some different files for.eg. DockerFile, etc. to run them on the cloud.
