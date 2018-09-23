import pickle
import preprocess
def change_tfidf(text):
    pre = preprocess.Preprocess()
    with open('tfidf.pickle', 'rb') as handle:
        vectorizer = pickle.load(handle)
    clean = ' '.join(e for e in pre.preprocess(text))
    vector = vectorizer.transform([clean])
    return vector.toarray()

def predict(text):
    with open('model_knn.data', 'rb') as handle:
        model = pickle.load(handle)
    vector = change_tfidf(text)
    return model.predict_proba(vector)