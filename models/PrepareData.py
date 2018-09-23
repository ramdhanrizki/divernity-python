#%%
# Import Dataset
import pandas as pd
data = pd.read_csv("data/divernity_train.csv")
print('Data sara {}'.format(data[data['Label']=='Sara'].shape))
print('Data bukan sara {}'.format(data[data['Label']=='Tidak Sara'].shape))
#%%
artikel = data['Artikel '].values
label = data['Label'].values
import preprocess
pre = preprocess.Preprocess()
#%%
# Cleanning data
document = []
for text in artikel:
    document.append(pre.preprocess(text))

#%%
words = []
for i in document:
    words.extend(i)
#%%
# Train TfIdf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(words)

# Summary
print(vectorizer.vocabulary_)
print(vectorizer.idf_)

#%%
# Fungsi untuk mengubah text menjadi tfidf
def change_tfidf(text):
    vector = vectorizer.transform([text])
    return vector.toarray()

#%%
# Menyiapkan data latih dan train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(artikel, label, test_size=0.2, random_state=42)

X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)
training = X_train_vec.toarray()
test = X_test_vec.toarray()

#%%
# Model Klasifikasi
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(training, y_train)

#%%
# KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(training, y_train)
classifier.score(test, y_test)

import pickle
pickle.dump(classifier, open('model_knn.data','wb'))
#%%
from sklearn import svm
modelsvm = svm.SVC()
modelsvm.fit(training,y_train)
modelsvm.score(test, y_test)

def predict(model, text):
    vector = change_tfidf(text)
    return model.predict(vector)

#%%
import scipy.sparse as sp
import json
json.dump(vectorizer.vocabulary_, open('dictionary/vocabulary.json', mode ='wb'))

#%%
import pickle
with open('vocabulary.data','wb') as handle:
    pickle.dump(vectorizer.vocabulary_,handle)

#%%
import pickle
with open('tfidf.pickle','wb') as handle:
    pickle.dump(vectorizer,handle)


