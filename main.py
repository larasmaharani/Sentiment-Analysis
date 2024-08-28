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

# Define preprocessing functions
def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # remove hashtags
    text = re.sub(r"http\S+", '', text) # remove links
    text = re.sub(r'[0-9]+', '', text) # remove numbers
    text = text.replace('\n', ' ') # replace new line with space
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove all punctuations
    text = text.strip() # remove spaces from both ends
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

# Load tokenizer and model
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
    
    # Preprocess the text
    text = cleaningText(review)
    text = casefoldingText(text)
    text = tokenizingText(text)
    text = filteringText(text)
    text = stemmingText(text)
    text = toSentence(text)
    
    # Tokenize new data
    sequences = tokenizer.texts_to_sequences([text])
    
    # Padding sequences
    max_length = 100  # Adjust according to your model's max_length
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    
    # Perform prediction
    predictions = model.predict(padded_sequences)
    
    # Get the index of the maximum value in each prediction
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Define labels corresponding to indices
    labels = ['negatif', 'netral', 'positif']
    
    # Select the label based on prediction result
    predicted_sentiment = labels[predicted_labels[0]]
    
    return {"prediksi": predicted_sentiment}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)



# import pickle
# import numpy as np

# from fastapi import FastAPI
# import uvicorn
# from pydantic import BaseModel
# from typing import Union
# from pydantic import BaseModel

# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# app = FastAPI()

# class nameClass(BaseModel):
#     text: str

# @app.get("/")
# def read_root():
#     return {"message": "Hello, World!"}

# @app.post("/prediction")
# def prediction(ulasan:nameClass):
#     review = ulasan.text
#         # Load tokenizer's configuration
#     with open('./model/tokenizer_config_v1.pkl', 'rb') as f:
#         tokenizer_cxonfig = pickle.load(f)

#     # Recreate tokenizer
#     tokenizer = Tokenizer(**tokenizer_config)

#     # Load tokenizer's word index
#     with open('./model/tokenizer_word_index_v1.pkl', 'rb') as f:
#         tokenizer.word_index = pickle.load(f)

#     # ulasan = input("Your sentimen:")

#     # Example new data to predict
#     new_texts = [review]

#     # Tokenize new data
#     sequences = tokenizer.texts_to_sequences(new_texts)

#     # Padding sequences
#     max_length = 100 # Adjust according to your model's max_length
#     padded_sequences = pad_sequences(sequences, maxlen=max_length)

#     # Load the saved model
#     model = load_model('./model/model_lstm.h5')

#     # Perform predictio
#     predictions = model.predict(padded_sequences)

#     # Mengambil indeks nilai maksimum dari setiap prediksi
#     predicted_labels = np.argmax(predictions, axis=1)

#     # Definisikan label yang sesuai dengan indeks
#     labels = ['negative', 'neutral', 'positive']

#     # Memilih label berdasarkan hasil prediksi
#     predicted_sentiment = labels[predicted_labels[0]]

#     # # Menampilkan hasil prediksi
#     # print("Sentimen Prediksi:", predicted_sentiment)

#     # print(predictions)

#     return {"prediksi" : predicted_sentiment}