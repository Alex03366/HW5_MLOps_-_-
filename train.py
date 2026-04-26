import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris_classification")

with mlflow.start_run(run_name="logistic_regression"):
    mlflow.log_params(params['model'])
    
    model = LogisticRegression(**params['model'])
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", accuracy)
    
    with open('accuracy.txt', 'w') as f:
        f.write(str(accuracy))
    
    os.makedirs('models', exist_ok=True)
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    mlflow.sklearn.log_model(model, "model")
    
    print(f"✅ Accuracy: {accuracy:.4f}")
