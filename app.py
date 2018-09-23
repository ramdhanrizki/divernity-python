import os
import sys

import requests
from flask import Flask, request, jsonify, json

from preprocess import Preprocess
import models.KNN as model
import numpy as np

#################### MAIN FLASK APPS ##########################

app = Flask("__name__")


@app.route('/')
def hello():
    return "Selamat datang"

@app.route('/sara', methods=['POST'])
def prediksi():
    label = ["Terindikasi Sara","Bukan Sara"]
    if request.is_json:
        content = request.get_json()
        text = content['text']
    else:
        text = request.form['text']

    prediksi = model.predict(text)
    index = np.where(prediksi[0]==prediksi[0].max())
    return_json = {
        'status' : 200,
        'message' : 'success',
        'klasifikasi' : label[index[0][0]],
        'detail' :
            {
                label[0]:str(prediksi[0][0]),
                label[1]:str(prediksi[0][1])
            }
    }
    return jsonify(return_json)

if __name__ == "__main__":
    app.run(debug=True)
    app.config["JSON_SORT_KEYS"] = False

