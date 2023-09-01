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

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

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

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# creating model

i = Input(shape=(input_shape,))
x = Embedding(vocabulary+1,10)(i)
x = LSTM(10, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation="softmax")(x)
model = Model(i,x)

# compiling the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# # trainning the model
# train = model.fit(x_train, y_train, epochs=200)
# model.save('chatbotmodel.h5', train)

# Training the model
train = model.fit(x_train, y_train, epochs=200, validation_data=(x_test, y_test))
model.save('chatbotmodel.h5', train)

# Plotting accuracy vs loss diagrams for training and testing datasets
plt.figure(figsize=(12, 4))

# Plotting loss
plt.subplot(1, 2, 2)
plt.plot(train.history['loss'], label='Train')
plt.plot(train.history['val_loss'], label='Test')
plt.title('Model Loss - Training and Testing Data')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


# Plotting accuracy
plt.subplot(1, 2, 1)
plt.plot(train.history['accuracy'], label='Train')
plt.plot(train.history['val_accuracy'], label='Test')
plt.title('Model Accuracy - Training and Testing Data')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Generating confusion matrix and classification report for training dataset
train_pred = model.predict(x_train)
train_pred_labels = np.argmax(train_pred, axis=1)
conf_matrix_train = confusion_matrix(y_train, train_pred_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Training Dataset')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("Classification Report - Training Dataset:\n")
print(classification_report(y_train, train_pred_labels, target_names=le.classes_))

# Generating confusion matrix and classification report for testing dataset
test_pred = model.predict(x_test)
test_pred_labels = np.argmax(test_pred, axis=1)
conf_matrix_test = confusion_matrix(y_test, test_pred_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Testing Dataset')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("Classification Report - Testing Dataset:\n")
print(classification_report(y_test, test_pred_labels, target_names=le.classes_))
