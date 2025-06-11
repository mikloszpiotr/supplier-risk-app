import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    print(classification_report(y_val, model.predict(X_val)))
    return model

def save_model(model, path='models/classifier.pkl'):
    joblib.dump(model, path)
