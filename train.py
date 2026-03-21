import os
import sys
import argparse
import logging
from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, recall_score, precision_score, average_precision_score,make_scorer
from feature_engine.encoding import CountFrequencyEncoder
from xgboost import XGBClassifier


import skops.io as sio
import mlflow

import s3fs
import pandas as pd


from src.data.loader import load_data
from src.features.preprocess import preprocess, build_features
from src.models.evaluate import evaluate
from src.utils.config import load_config



logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.DEBUG,
    handlers=[logging.FileHandler("recording.log"), logging.StreamHandler()],
)



# ENVIRONMENT CONFIGURATION ---------------------------

parser = argparse.ArgumentParser(
    description="Paramètres du XGBoost + grid search"
)
parser.add_argument(
    "--experiment_name", type=str, default="wildfire_ML", help="MLFlow experiment name"
)
parser.add_argument(
    "--n_estimators",
    type=int,
    default=300,
    help="Valeur par défaut pour n_estimators dans la grille",
)
parser.add_argument(
    "--cv", type=int, default=5, help="Nombre de folds pour la cross-validation"
)
args = parser.parse_args()

n_estimators_default = args.n_estimators
cv_folds = args.cv

logging.debug(f"Valeur de l'argument n_trees: {n_estimators_default}")
logging.debug(f"Valeur de l'argument cv: {cv_folds}")

# LOGGING IN MLFLOW -----------------

mlflow_server = os.getenv("https://user-hugoseumen-mlflow.user.lab.sspcloud.fr/#/experiments")

logging.debug(f"Saving experiment in {mlflow_server}")

mlflow.set_tracking_uri(mlflow_server)
mlflow.set_experiment(args.experiment_name)


# LOADING DATA  + PREPROCESSING  -----------------------------------------

df_raw = pd.read_csv("https://minio.lab.sspcloud.fr/hugoseumen/wildfire-mlops/data/raw_data.csv")
logging.info("Chargement des données ✅")
logging.info(df_raw.head(5))
sys.exit(0)

df_clean = preprocess(df_raw)


target_col = "ignition"
n_neg = (df_clean[target_col] == 0).sum()
n_pos = (df_clean[target_col] == 1).sum()
pos_weight = n_neg / n_pos
logging.info(f"pos_weight: {pos_weight:.2f}") 

X_train, X_test, y_train, y_test, preprocessor, feature_names = build_features(
    df_clean, target_col=target_col, encoding="onehot"
)

#  RANDOMIZED SEARCH CV  -----------------------------------------

param_dist = {
        "n_estimators": [300, 500, 700],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth":  [3, 4, 5, 6, 7],
        "subsample":  [0.6, 0.8, 1.0],
        "colsample_bytree":  [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 1, 3]
    }


train_data = pd.concat([X_train, y_train], axis=1)

with mlflow.start_run():

    logging.debug(f"\n{80 * '-'}\nLogging input in MLFlow\n{80 * '-'}")

    mlflow.log_input(
        mlflow.data.from_pandas(df_raw),
        context="raw",
    )

    mlflow.log_input(
        mlflow.data.from_pandas(train_data),
        context="raw",
    )

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=pos_weight
    )

    scorer = make_scorer(average_precision_score, response_method='predict_proba')

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        scoring=scorer,
        cv=5,
        verbose=0,
        refit=True,
    )

# TRAINING AND EVALUATION --------------------------------------------

    logging.debug(f"\n{80 * '-'}\nStarting randomized search fitting phase\n{80 * '-'}")

    search.fit(X_train, y_train)

    logging.info(f"Best CV score: {search.best_score_:.3f}")
    logging.info(f"Best params: {search.best_params_}")

    best_model = search.best_estimator_

    best_params = search.best_params_

    for param, value in best_params.items():
        mlflow.log_param(param, value)


# Sauvegarde du meilleur pipeline complet
    sio.dump(best_model, "model.skops")

# Évaluation propre avec les bonnes métriques
    test_metrics = evaluate(best_model, X_test, y_test)
    train_metrics = evaluate(best_model, X_train, y_train)

    logging.info(f"Test PR-AUC: {test_metrics["pr_auc"]}")
    logging.info(f"Train PR-AUC: {train_metrics["pr_auc"]}") # Il y a du overfit 
    

# Log metrics
    mlflow.log_metric("PR-AUC", test_metrics["pr_auc"])

    logging.debug(f"\n{80 * '-'}\nFILE ENDED SUCCESSFULLY!\n{80 * '-'}")

 # Log model
    mlflow.sklearn.log_model(xgb, "model")


