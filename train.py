import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle


X = pd.read_csv("processed_data.csv").values
y = pd.read_csv("labels.csv").values.flatten()


X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)


with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


class DiseasePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DiseasePredictor, self).__init__()
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


input_size = X.shape[1]  
hidden_size = 128
output_size = len(np.unique(y))  

model = DiseasePredictor(input_size, hidden_size, output_size)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Save trained model
torch.save(model.state_dict(), "disease_model.pth")

print("✅ Model trained and saved!")
