import joblib
import pandas as pd

MODEL_PATH = 'models/classifier.pkl'
ENCODER_PATH = 'models/label_encoder.pkl'

model = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

def predict_risk(df: pd.DataFrame):
    proba = model.predict_proba(df)
    preds = proba.argmax(axis=1)
    risk_levels = le.inverse_transform(preds)
    return risk_levels

def get_feature_importance(feature_names):
    return pd.Series(model.feature_importances_, index=feature_names)
