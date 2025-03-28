from flask import Flask, render_template, request
import pickle
import torch
import pandas as pd
import torch.nn as nn

app = Flask(_name_)


with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


data = pd.read_csv("MedicalDataset.csv")

class DiseasePredictor(nn.Module):
    def _init_(self, input_size, hidden_size, output_size):
        super(DiseasePredictor, self)._init_()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


input_size = len(vectorizer.get_feature_names_out())
hidden_size = 128
output_size = len(label_encoder.classes_)
model = DiseasePredictor(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("disease_model.pth"))
model.eval()


def predict_disease(symptom_input):
    symptom_vector = vectorizer.transform([symptom_input]).toarray()
    symptom_tensor = torch.tensor(symptom_vector, dtype=torch.float32)

    output = model(symptom_tensor)
    predicted_class = torch.argmax(output, dim=1).item()
    predicted_disease = label_encoder.inverse_transform([predicted_class])[0]

    test_info = data[data["Disease"] == predicted_disease]
    if not test_info.empty:
        recommended_tests = test_info["Medical Tests"].values[0]
        test_price = test_info["Cost (?)"].values[0]
    else:
        recommended_tests, test_price = "Not Available", "Unknown"

    return predicted_disease, recommended_tests, test_price


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    user_symptoms = request.form['symptoms']
    disease, tests, cost = predict_disease(user_symptoms)

    return render_template('result.html', disease=disease, tests=tests, cost=cost)

if _name_ == '_main_':
    app.run(debug=True)
