#%%
# Import Dataset
words = []
classes = []
document = []
import pandas as pd
data = pd.read_excel("data/comrades_mitos_fakta.xlsx")
# print('Data mitos {}'.format(data[data['Label']=='Mitos'].shape))
# print('Data fakta {}'.format(data[data['Label']=='Fakta'].shape))


#%%
# Prepare Train Data
# import preprocess
# import nltk
# pre = preprocess.Preprocess()
# for index, row in data.iterrows():
#     w = nltk.word_tokenize(row['Text'])
#     words.extend(w)
#     document.append((w, row['Label']))
#     if row['Label'] not in classes:
#         classes.append(row['Label'])
# print('Proses Selesai')

#%%
# Prepare Train Data
import preprocess
pre = preprocess.Preprocess()
for index, row in data.iterrows():
    w = pre.preprocess(row['text'])
    words.extend(w)
    document.append((w, row['label']))
    if row['label'] not in classes:
        classes.append(row['label'])
print('Proses Selesai')

# Menghilangkan duplikat words dan class
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(len(document), " documents")
print(len(classes), "classes ", classes)
print(len(words), " unique stemmed words", words)

#%%
# Membuat model tfidf
training = []
output = []
output_empty = [0] * len(classes)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(words)

for doc in document:
    pattern_words = doc[0]
    vector = vectorizer.transform(pattern_words)
    print(vector.toarray())

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([vector.toarray()[0], output_row])

#%%
# TF IDF
def prepare_data(text):
    pre = preprocess.Preprocess()
    with open('tfidf_comrades2.pickle', 'rb') as handle:
        vectorizer = pickle.load(handle)
    clean = ' '.join(e for e in pre.preprocess(text))
    vector = vectorizer.transform([clean])
    return vector.toarray()
#%%
# Random data train
import random
import numpy as np
random.shuffle(training)
training = np.array(training)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

x_train = vectorizer.transform(X_train)
x_test = vectorizer.transform(X_test)

train_x = list(training[:,0])
train_y = list(training[:,1])

#%%
# Bikin Model
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import utils
from keras import layers

model = Sequential()
model.add(Dense(25, input_shape=[len(train_x[0])]))
model.add(Dense(25, activation='sigmoid'))
model.add(Dense(25, activation='sigmoid'))
model.add(Dense(len(train_y[0]), activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=8)

#%%
# Simpan Model Klasifikasi
import matplotlib.pyplot as plt
model.save('model_mitosfakta2.h5')
history_dict = history.history
history_dict.keys()
acc = history.history['acc']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#%%
import pickle
with open('tfidf_comrades2.pickle','wb') as handle:
    pickle.dump(vectorizer,handle)


#%%
# Coba KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(x_train.toarray(0), y_train)

import pickle
pickle.dump(classifier, open('knn_comrades.data','wb'))