"""Training pipeline for wildfire ignition classification."""

import os
import argparse
import logging
import requests

from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, make_scorer
from xgboost import XGBClassifier

import joblib
import skops.io as sio
import mlflow
from mlflow.models import infer_signature
from dotenv import load_dotenv

import pandas as pd

from src.features.preprocess import preprocess, build_features
from src.models.evaluate import evaluate

# 1. CHARGEMENT DES VARIABLES D'ENVIRONNEMENT ---------------------------
load_dotenv()

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.DEBUG,
    handlers=[logging.FileHandler("recording.log"), logging.StreamHandler()],
)

# 2. CONFIGURATION DES ARGUMENTS ---------------------------------------
# 2. CONFIGURATION DES ARGUMENTS ---------------------------------------
parser = argparse.ArgumentParser(description="Paramètres du XGBoost + MLflow")

# Configuration de l'environnement
parser.add_argument(
    "--mode",
    type=str,
    choices=["shared", "personal"],
    default="shared",
    help="Choisir entre le service partagé (par défaut) ou votre service personnel"
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="wildfire_ML",
    help="Nom de l'expérience MLflow"
)

# Hyperparamètres XGBoost
parser.add_argument(
    "--n_estimators",
    type=int,
    default=300,
    help="Nombre d'arbres pour XGBoost"
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.05,
    help="Pas d'apprentissage (vitesse de convergence)"
)
parser.add_argument(
    "--max_depth",
    type=int,
    default=5,
    help="Profondeur maximale des arbres"
)
parser.add_argument(
    "--subsample",
    type=float,
    default=0.8,
    help="Fraction des données d'entraînement utilisée par arbre"
)
parser.add_argument(
    "--colsample_bytree",
    type=float,
    default=0.8,
    help="Fraction des colonnes (features) utilisée par arbre"
)

# Configuration de la validation
parser.add_argument(
    "--cv",
    type=int,
    default=5,
    help="Nombre de folds pour la cross-validation"
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=10,
    help="Nombre d'itérations pour la recherche aléatoire (Random Search)"
)

args = parser.parse_args()

# 3. LOGIQUE DE CONNEXION MLFLOW ---------------------------------------
SHARED_URI = "https://user-gb53-mlflow.user.lab.sspcloud.fr"

if args.mode == "personal":
    mlflow_server = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_server:
        logging.error("Mode personnel choisi mais MLFLOW_TRACKING_URI est absente du .env")
        exit(1)
    logging.info("Utilisation de votre instance MLflow personnelle.")
else:
    mlflow_server = os.getenv("MLFLOW_TRACKING_URI", SHARED_URI)
    logging.info(f"Utilisation de l'instance partagée : {mlflow_server}")

mlflow_server = mlflow_server.split("/#")[0].rstrip("/")

try:
    requests.get(mlflow_server, timeout=5)
    mlflow.set_tracking_uri(mlflow_server)
    mlflow.set_experiment(args.experiment_name)
    logging.info("Connexion au serveur MLflow établie.")
except Exception as e:
    logging.error(f"Serveur MLflow injoignable : {e}")
    exit(1)

mlflow.sklearn.autolog(log_datasets=False, silent=True)

# 4. CHARGEMENT ET PRÉPARATION DES DONNÉES -----------------------------
logging.info("Chargement des données depuis MinIO...")
df_raw = pd.read_csv("https://minio.lab.sspcloud.fr/hugoseumen/wildfire-mlops/data/raw_data.csv")

df_clean = preprocess(df_raw)

target_col = "ignition"
pos_weight = (df_clean[target_col] == 0).sum() / (df_clean[target_col] == 1).sum()

X_train, X_test, y_train, y_test, preprocessor = build_features(
    df_clean, target_col=target_col, encoding="onehot"
)

# 5. ENTRAÎNEMENT ET TRACKING -----------------------------------------
run_name = f"xgb_{args.mode}_cv{args.cv}"

with mlflow.start_run(run_name=run_name):
    mlflow.log_input(mlflow.data.from_pandas(df_raw), context="raw")

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=pos_weight,
        random_state=42,
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", xgb)])

    param_dist = {
        "classifier__n_estimators": [args.n_estimators, args.n_estimators + 200],
        "classifier__learning_rate": [args.learning_rate, args.learning_rate / 2],
        "classifier__max_depth": [args.max_depth, args.max_depth + 2],
        "classifier__subsample": [args.subsample, 1.0],
        "classifier__colsample_bytree": [args.colsample_bytree, 1.0],
    }

    scorer = make_scorer(average_precision_score, response_method="predict_proba")

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        scoring=scorer,
        cv=args.cv,
        refit=True,
        n_iter=10
    )

    logging.info("Running Randomized Search...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # 6. SIGNATURE ET EVALUATION ---------------------------------------
    signature = infer_signature(X_test, best_model.predict(X_test))

    test_metrics = evaluate(best_model, X_test, y_test)
    for metric, value in test_metrics.items():
        mlflow.log_metric(f"test_{metric}", value)

        # 7. SAUVEGARDE LOCALE ET REGISTRE ---------------------------------
        backup_dir = "models_backup"
        os.makedirs(backup_dir, exist_ok=True)

        # Définition des chemins
        model_path = os.path.join(backup_dir, "model.skops")
        preprocessor_path = os.path.join(backup_dir, "preprocessor.pkl")
        cv_results_path = os.path.join(backup_dir, "cv_results.csv")

        # Sauvegarde Locale en Baclup
        sio.dump(best_model, model_path)
        joblib.dump(preprocessor, preprocessor_path)

        # Sauvegarde du tableau complet des résultats de recherche
        cv_results_df = pd.DataFrame(search.cv_results_)
        cv_results_df.to_csv(cv_results_path, index=False)

        # Envoi des fichiers locaux vers MLflow (Artifacts)
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(preprocessor_path)
        mlflow.log_artifact(cv_results_path)

        # Enregistrement officiel dans le Model Registry
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            registered_model_name="WildfireXGBClassifier"
        )

        logging.info(f"Meilleur score CV : {search.best_score_:.3f}")
        logging.info(f"Tableau cv_results enregistré localement : {cv_results_path}")
        logging.info(f"Backup complet effectué dans le dossier : {backup_dir}/")
        logging.info(f"Modèle enregistré dans le registre : WildfireXGBClassifier")
        logging.debug("Script terminé avec succès.")