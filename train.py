import numpy as np
import random
import json
import pandas as pd
import string
from keras.models import Sequential, model_from_json 
from keras.layers import * 
from keras.models import Model   
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open('sample.json', 'r') as f:
    intents = json.load(f)
    
# getting all data into lists
tags = []
inputs = []
responses = {}

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    responses[intent['tag']] = intent['responses']
    for pattern in intent['patterns']:
        inputs.append(pattern)
        # add to xy pair
        tags.append(intent['tag'])
        
data = pd.DataFrame({"inputs":inputs, "tags":tags})

# removing puncuations
data['inputs'] = data['inputs'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))

pickle.dump(inputs, open('words.pkl','wb'))
pickle.dump(tags, open('tags.pkl','wb'))

# tokenize the data
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])

# apply padding
from keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(train)

# encoding the output
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(data.tags)

input_shape = x_train.shape[1]
print(input_shape)

# define vocab
vocabulary = len(tokenizer.word_index)
print("number of unique words :", vocabulary)

output_length = le.classes_.shape[0]
print("output_length :", output_length)

# creating model

i = Input(shape=(input_shape,))
x = Embedding(vocabulary+1,10)(i)
x = LSTM(10, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation="softmax")(x)
model = Model(i,x)

# compiling the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# trainning the model
train = model.fit(x_train, y_train, epochs=200)
model.save('chatbotmodel.h5', train)

# evaluate the model
scores = model.evaluate(x_train, y_train, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# model analysis
plt.plot(train.history['accuracy'], label="trainning set accuracy")
plt.plot(train.history['loss'], label="trainning set loss")
plt.legend()
plt.tight_layout()
plt.show()

# data = {
# "input_shape": input_shape,
# "output_length": output_length,
# "vocabulary": vocabulary,
# "responses": responses,
# "inputs": inputs,
# "tags": tags,
# "le":le
# }

# FILE = "data.pth"
# torch.save(data, FILE)

# while True:
#     texts_p = []
#     prediction_input = input("You : ")
    
#     # removing puncuation
#     prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
#     prediction_input = ''.join(prediction_input)
#     texts_p.append(prediction_input)
    
#     # tokenizing and padding
#     prediction_input = tokenizer.texts_to_sequences(texts_p)
#     prediction_input = np.array(prediction_input).reshape(-1)
#     prediction_input = pad_sequences([prediction_input], input_shape)
    
#     # getting an output from model
#     output = model.predict(prediction_input)
#     output = output.argmax()
    
#     response_tag = le.inverse_transform([output])[0]
#     print("Bim : ", random.choice(responses[response_tag]))
    
#     if response_tag == 'quit':
#         break