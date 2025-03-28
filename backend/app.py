from flask import Flask, request, jsonify, render_template
import numpy as np
from model import predict_disease  # Importing LSTM prediction function

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data.get('symptoms', '')

    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400

    disease, test, cost = predict_disease(symptoms)
    
    return jsonify({'disease': disease, 'test': test, 'cost': cost})

if __name__ == '__main__':
    app.run(debug=True)
