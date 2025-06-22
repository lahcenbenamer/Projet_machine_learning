from flask import Flask, render_template, request
import pickle
import pandas as pd
import json

# Classe personnalisée avec seuil ajustable
class ThresholdModelWrapper:
    def __init__(self, model, threshold=0.4):
        self.model = model
        self.threshold = threshold

    def predict(self, X):
        return (self.model.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

# Initialisation de l'app Flask
app = Flask(__name__)

# Chargement du modèle, du scaler et des noms de colonnes
with open("model/model.pkl", "rb") as f:
    base_model, scaler, feature_names = pickle.load(f)
    model = ThresholdModelWrapper(base_model, threshold=0.4)

# Chargement des scores depuis le fichier JSON
try:
    with open("data/model_scores.json", "r") as f:
        model_scores = json.load(f)
except FileNotFoundError:
    model_scores = {}

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    show_metrics = False

    if request.method == "POST":
        if "predict" in request.form:
            user_input = request.form.to_dict()
            input_df = pd.DataFrame([user_input])
            input_df = pd.get_dummies(input_df)
            input_df = input_df.reindex(columns=feature_names, fill_value=0)
            input_scaled = scaler.transform(input_df)

            prediction_proba = model.predict_proba(input_scaled)[0][1]
            prediction = model.predict(input_scaled)[0]
            result = (
                f"Client en DEF AUT de paiement (probabilité = {prediction_proba:.2f})"
                if prediction == 1 else
                f"Client NON défaillant (probabilité = {prediction_proba:.2f})"
            )

        elif "metrics" in request.form:
            show_metrics = True
            result = None  # On vide le message de prédiction si on demande juste les performances

    return render_template("index.html", result=result, show_metrics=show_metrics, model_scores=model_scores)

if __name__ == "__main__":
    app.run(debug=True)
