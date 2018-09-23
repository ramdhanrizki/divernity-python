import re
from nltk.tokenize import TweetTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import dictionary
import json
import requests

class Preprocess:
    def __init__(self, preprocessing_dataset=None):
        self.tokenizer = TweetTokenizer()
        self.stop_words = dictionary.get_stop_words()
        self.base_words = dictionary.get_base_words()
        self.slang_words = dictionary.get_slang_words()
        self.stemmer = StemmerFactory().create_stemmer()
        self.preprocessing_dataset = preprocessing_dataset

    def case_folding(self, document):
        document = document.lower()
        return document

    def clean(self, document):
        document = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', '', document)  # URLs
        document = re.sub(r'RT', '', document)  # Retweet
        document = re.sub(r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)', '', document)  # Hashtag
        document = re.sub(r'(?:@[\w_]+)', '', document)  # Mention
        document = re.sub(r'[^\x00-\x7F]+', '', document)  # Unicode
        document = re.sub(r'rt', '', document)  # Retweet
        document = re.sub(r'(?:[:=;][oO\-]?[D\)\]\(\]/\\OpP])', '', document)  # Emoticon
        document = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)', '', document)  # Special Char
        document = re.sub(r'[\n\t\r]+', '', document) # Remove linebreak, tab, return
        return document

    def tokenize(self, document):
        tokenized_document = self.tokenizer.tokenize(document)
        return tokenized_document

    def stopword_removal(self, document):
        document = [word for word in document if word not in self.stop_words]
        return document

    def stem(self, document):
        document = self.stemmer.stem(document)
        return document

    def slang_word_correction(self, document):
        pattern = re.compile(r'\b(' + '|'.join(self.slang_words.keys()) + r')\b')
        document = pattern.sub(lambda x: self.slang_words[x.group()], document)
        return document

    def base_word_check(self, document):
        document = [word for word in document if word in self.base_words]
        return document

    def preprocess(self, document):
        document = self.case_folding(document)
        document = self.clean(document)
        document = self.slang_word_correction(document)
        document = self.stem(document)
        document = re.sub(r'[^a-zA-Z ]', '', document)  # Special Char
        document = self.tokenize(document)
        document = self.stopword_removal(document)
        document = self.base_word_check(document)
        return document

    def list_to_string(self, document):
        str = ' '.join(document)
        return str

    def save_preprocessed_text(self, file_name, document):
        with open(file_name, 'w') as outfile:
            json.dump(document, outfile, indent=4)

        url = 'https://unikom-sentiment-services.azurewebsites.net/upload-ps'
        files = {'json': open(file_name, 'rb')}
        request = requests.post(url, files=files)
        if(request.status_code == 200):
            return True

        return False

    def load_preprocessed_text(self):
        collection = self.preprocessing_dataset.find({})
        if collection.count() == 0:
            return False
        
        for preprocessing_data in collection:
            words_vocabulary = preprocessing_data['data'][0]
            classes = preprocessing_data['data'][1]

        return words_vocabulary, classes