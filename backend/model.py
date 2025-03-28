import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("dataset.csv")  # Ensure this file is in the same directory

# Tokenization
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['Symptoms'])

# Encoding Labels
label_encoder_disease = LabelEncoder()
label_encoder_test = LabelEncoder()

data['Disease'] = label_encoder_disease.fit_transform(data['Disease'])
data['Medical Tests'] = label_encoder_test.fit_transform(data['Medical Tests'])

# Load trained model (Train separately)
model = load_model("lstm_model.h5")

def predict_disease(symptoms):
    sequence = tokenizer.texts_to_sequences([symptoms])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')

    prediction = model.predict(padded_sequence)
    
    disease_index = np.argmax(prediction, axis=1)[0]
    disease_predicted = label_encoder_disease.inverse_transform([disease_index])[0]
    
    test_predicted = data[data['Disease'] == disease_predicted]['Medical Tests'].values[0]
    cost_predicted = data[data['Disease'] == disease_predicted]['Cost (₹)'].values[0]
    
    return disease_predicted, test_predicted, cost_predicted
