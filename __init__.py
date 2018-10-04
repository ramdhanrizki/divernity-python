import os
import sys

import requests
from flask import Flask, request, jsonify

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
nltk.download('punkt')
nltk.download('stopwords')
from wordcloud import WordCloud,STOPWORDS
#import matplotlib.pyplot as plt
from .Preprocessing import Preprocess

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from tensorflow import keras
from keras.models import load_model
import json

#from pathlib import Path
## Load Dataset
with open(os.path.realpath('var/www/Comrades/Comrades/data/comrades_intent.json')) as json_data:
    intents = json.load(json_data)
    
# Masukkan dataset ke dalam array 
words = []
classes = []
documents = []
ignore_words = ['?']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates
classes = sorted(list(set(classes)))

#print (len(documents), "documents")
#print (len(classes), "classes", classes)
#print (len(words), "unique stemmed words", words)

model = load_model(os.path.realpath('var/www/Comrades/Comrades/data/model_Comrades.h5'))

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

context = {}    
ERROR_THRESHOLD = 0.65
def classify(sentence):
    # generate probabilities from the model
    p = bow(sentence, words)
    
    d = len(p)
    f = len(documents)-2
    a = np.zeros([f, d])
    tot = np.vstack((p,a))
    
    #results = model.predict([bow(sentence, words)])[0]
    results = model.predict(tot)[0]
    
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list


def response(sentence, userID=123, show_details=False):
    results = classify(sentence)
    hasil = []
    print('Result:',results)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    hasil.append({"context":i['tag'],"score":str(results[0][1]),"response":random.choice(i['responses'])})
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    #if not 'context_filter' in i or \
                    #    (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                    #    if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                    #    return (random.choice(i['responses']))
            results.pop(0)
        return hasil

import pickle 

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

sentiment_classes = ['Positive','Negative']
output_sentiment = []
output_empty_sentiment = [0] * len(sentiment_classes)

with open('var/www/Comrades/Comrades/data/training4.data','rb') as f:
        data,documents,words = pickle.load(f)

def bow_en(sentence, words, show_details=False):
    # tokenize the pattern
    pre = Preprocess()
    sentence_words = pre.preprocess(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))



def pred(sentence,syn_01, syn_12, syn_23):
        data = bow_en(sentence,words,show_details=False)
        for syn in [syn_01, syn_12, syn_23]:
            data = sigmoid(np.dot(data, syn))
        hasil = [[i,r] for i,r in enumerate(data)] 
        hasil.sort(key=lambda x: x[1], reverse=True) 
        hasil_akhir =[[sentiment_classes[r[0]],r[1]] for r in hasil]
        return hasil_akhir

#################### MAIN FLASK APPS ##########################

app = Flask("__name__")
@app.route('/')
def hello():
    return "Hello World"

@app.route('/chatbot', methods=['POST'])
def chatbot():
    if request.is_json:
        content = request.get_json()
        text = content['text']
        userID = content['userid']
    else:
        text = request.form['text']
        userID = request.form['userid']
    
    return jsonify({"status":200,"message":"success","result":response(text,userID)})

@app.route('/sentiment/en', methods=['post'])
def sentiment_en():
    if request.is_json:
        content = request.get_json()
        text = content['text']
    else:
        text = request.form['text']

    ## LOAD DATA ##
    

    with open(os.path.realpath('var/www/Comrades/Comrades/data/synapse1.data'),'rb') as j:
        syn_01,syn_12,syn_23,training_sentiment,output_sentiment = pickle.load(j)
    ## END OF LOAD DATA ##

    hasil = pred(text, syn_01, syn_12, syn_23)
    #return hasil[0][0]
    return jsonify({"status":200,"message":"success","result":hasil})

if __name__ == "__main__":
    app.run(debug=False)
    app.config["JSON_SORT_KEYS"] = False

