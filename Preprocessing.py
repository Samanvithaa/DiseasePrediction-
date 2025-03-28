import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("MedicalDataset.csv")


if not all(col in df.columns for col in ["Disease", "Symptoms", "Medical Tests", "Cost (?)"]):
    raise ValueError("Dataset must have columns: Disease, Symptoms, Medical Tests, Cost (?)")


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Symptoms"]).toarray()


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Disease"])


pd.DataFrame(X).to_csv("processed_data.csv", index=False)
pd.DataFrame(y, columns=["Disease"]).to_csv("labels.csv", index=False)


with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Preprocessing complete. Data saved!")
