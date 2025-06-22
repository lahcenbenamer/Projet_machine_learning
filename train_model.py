import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import os

# Chargement des données
df = pd.read_csv("../data/UCI_Credit_Card.csv")

# Nettoyage
df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
df['avg_bill'] = df[[f'BILL_AMT{i}' for i in range(1, 7)]].mean(axis=1)
df['avg_pay'] = df[[f'PAY_AMT{i}' for i in range(1, 7)]].mean(axis=1)
df.drop(columns=['ID'] + [f'BILL_AMT{i}' for i in range(1, 7)] + [f'PAY_AMT{i}' for i in range(1, 7)], inplace=True)

# Encodage
df = pd.get_dummies(df, columns=['SEX', 'EDUCATION', 'MARRIAGE'])

# Features et target
X = df.drop('default.payment.next.month', axis=1)
y = df['default.payment.next.month']

# Équilibrage avec SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)

# Standardisation
scaler = StandardScaler()
X_res_scaled = scaler.fit_transform(X_res)

# Division
X_train, X_test, y_train, y_test = train_test_split(X_res_scaled, y_res, stratify=y_res, test_size=0.25, random_state=42)

# Modèles
models = {
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    scores[name] = {
        'accuracy': round(report['accuracy'], 4),
        'precision': round(report['0']['precision'], 4),
        'recall': round(report['0']['recall'], 4),
        'f1_score': round(report['0']['f1-score'], 4)
    }
    print(f"\n{name}\n", classification_report(y_test, y_pred))

# Sauvegarde des scores
os.makedirs("../data", exist_ok=True)
with open("../data/model_scores.json", "w") as f:
    json.dump(scores, f, indent=4)

# Graphique
df_scores = pd.DataFrame(scores).T
df_scores.plot(kind='bar', figsize=(10, 6))
plt.title("Comparaison des modèles ")
plt.ylabel("Score")
os.makedirs("../static", exist_ok=True)
plt.tight_layout()
plt.savefig("../static/comparaison_modeles_scores.png")

# Sauvegarde du meilleur modèle (pas de ThresholdWrapper ici)
best_model = models['XGBoost']
with open("model.pkl", "wb") as f:
    pickle.dump((best_model, scaler, X.columns), f)
