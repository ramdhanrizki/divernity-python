import pickle
import preprocess

import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import utils
from keras import layers
from keras.models import load_model

def prepare_data(text):
    pre = preprocess.Preprocess()
    with open('tfidf_comrades2.pickle', 'rb') as handle:
        vectorizer = pickle.load(handle)
    clean = ' '.join(e for e in pre.preprocess(text))
    vector = vectorizer.transform([clean])
    return vector.toarray()

def klasifikasi(text):
    label = ['Mitos','Fakta']
    model = load_model('model_mitosfakta2.h5')
    vector = prepare_data(text)
    results = model.predict(vector[0])
    index = np.argmax(results)
    return label[index]