import re
import string
import pickle
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

class NameClass(BaseModel):
    text: str

def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r"http\S+", '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

def casefoldingText(text):
    return text.lower()

def tokenizingText(text):
    return word_tokenize(text)

def filteringText(text):
    listStopwords = set(stopwords.words('indonesian'))
    important_words = {"tidak", "ngga", "engga", "ga", "baik", "bagus", "tepat", "waktu", "masalah", "jangan"}
    return [word for word in text if word not in listStopwords or word in important_words]

def stemmingText(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return [stemmer.stem(word) for word in text]

def toSentence(list_words):
    return ' '.join(list_words)

with open('./model/tokenizer_config_v7.pkl', 'rb') as f:
    tokenizer_config = pickle.load(f)
tokenizer = Tokenizer(**tokenizer_config)
with open('./model/tokenizer_word_index_v7.pkl', 'rb') as f:
    tokenizer.word_index = pickle.load(f)
model = load_model('./model/model_lstm_v7.h5')

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/prediction")
def preprocess_and_predict(ulasan: NameClass):
    review = ulasan.text
    text = cleaningText(review)
    text = casefoldingText(text)
    text = tokenizingText(text)
    text = filteringText(text)
    text = stemmingText(text)
    text = toSentence(text)
    sequences = tokenizer.texts_to_sequences([text])
    max_length = 100
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    predictions = model.predict(padded_sequences)
    predicted_labels = np.argmax(predictions, axis=1)
    labels = ['negatif', 'netral', 'positif']
    predicted_sentiment = labels[predicted_labels[0]]
    return {"prediksi": predicted_sentiment}
