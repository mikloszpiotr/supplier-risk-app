import os
import pandas as pd
from src.preprocessing import load_raw, engineer_features, scale_features
from src.modeling import train_model, save_model
from sklearn.preprocessing import LabelEncoder
import joblib

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Load raw data
df = load_raw('data/raw/suppliers.csv')

# Feature engineering
features = engineer_features(df)
X, _ = scale_features(features)

# Target
y = df['risk_label']
# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train & save model
model = train_model(X, y_encoded)
save_model(model, path='models/classifier.pkl')

# Save label encoder
joblib.dump(le, 'models/label_encoder.pkl')

print("Model trained and saved to models/classifier.pkl and label_encoder.pkl")
