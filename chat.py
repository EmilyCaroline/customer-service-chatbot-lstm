import json
import pickle
import string
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import *
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import random

# Load intents from JSON
with open('sample.json', 'r') as f:
    intents = json.load(f)

# Extract data from intents
tags = []
inputs = []
responses = {}

for intent in intents['intents']:
    responses[intent['tag']] = intent['responses']
    for pattern in intent['patterns']:
        inputs.append(pattern)
        tags.append(intent['tag'])

data = pd.DataFrame({"inputs": inputs, "tags": tags})

# Preprocess text data
data['inputs'] = data['inputs'].apply(lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))

# Tokenize the data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])
x_train = pad_sequences(train)

# Encode the output labels
le = LabelEncoder()
y_train = le.fit_transform(data.tags)

input_shape = x_train.shape[1]
vocabulary = len(tokenizer.word_index)
output_length = le.classes_.shape[0]

# Load the pre-trained model
model = load_model('chatbotmodel.h5')

# Chatbot interaction loop
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    
    user_input = [user_input]
    user_input = tokenizer.texts_to_sequences(user_input)
    user_input = pad_sequences(user_input, maxlen=input_shape)
    
    prediction = model.predict(user_input)
    predicted_tag = le.inverse_transform([np.argmax(prediction)])
    response = random.choice(responses[predicted_tag[0]])
    
    print("Sam:", response)

