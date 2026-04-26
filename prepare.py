import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
import os

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

df = pd.read_csv('data/raw/iris.csv')
X = df.drop('variety', axis=1)
y = df['variety']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params['data']['test_size'],
    random_state=params['data']['random_state'], stratify=y
)

os.makedirs('data/processed', exist_ok=True)
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print(f"✅ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
