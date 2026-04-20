# 🔥 Wildfire Ignition Point Prediction — MLOps Project
![Wildfire](https://minio.lab.sspcloud.fr/hugoseumen/wildfire-mlops/static/wildfire.png)

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?logo=xgboost)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)
![ArgoCD](https://img.shields.io/badge/GitOps-ArgoCD-red?logo=argo)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue?logo=mlflow)
> Prédiction de points d'ignition de feux de forêt via un pipeline MLOps complet : de la donnée brute au déploiement en production.

**🌐 API en production** → [https://wildfire-api.user.lab.sspcloud.fr/docs](https://wildfire-api.user.lab.sspcloud.fr/docs)

---

## 📌 Contexte

Le point d'ignition est l'endroit où un incendie démarre. Prédire la probabilité qu'un point géographique soit un point d'ignition est essentiel pour mieux simuler et anticiper les feux de forêt.

Ce projet transforme un problème de **classification binaire** (ignition / pas d'ignition) en un produit ML industrialisé, déployé sur le [SSP Cloud](https://datalab.sspcloud.fr/).

---

## 🎯 Objectif

Construire un pipeline MLOps complet :

```
Données publiques (S3)
    → Preprocessing
    → Entraînement + Tuning (XGBoost + RandomizedSearchCV)
    → Tracking (MLflow)
    → API de prédiction (FastAPI)
    → Containerisation (Docker)
    → Déploiement cloud (Kubernetes / SSP Cloud)
    → GitOps (ArgoCD)
```

---

## 🗂️ Structure du projet

```
projet_meppds/
├── app/
│   └── api.py              # API FastAPI
├── src/
│   ├── data/
│   │   └── loader.py       # Chargement des données
│   ├── features/
│   │   └── preprocess.py   # Preprocessing + feature engineering
│   ├── models/
│   │   └── evaluate.py     # Métriques d'évaluation
│   └── utils/
│       └── config.py       # Chargement de la configuration
├── notebooks/
│   └── code.ipynb          # Exploration et analyse
├── .github/
│   └── workflows/
│       └── docker.yml      # CI/CD GitHub Actions
├── train.py                # Script d'entraînement principal
├── pyproject.toml          # Dépendances (uv)
├── Dockerfile              # Image Docker de l'API
├── .gitignore
├── LICENSE
└── README.md
```

> ⚙️ Le déploiement Kubernetes est géré dans un dépôt séparé : [application-deployment](https://github.com/hugzweek/application-deployment)

---

## 📊 Données

Chaque observation correspond à un point géographique avec les features suivantes :

| Catégorie | Features |
|-----------|----------|
| 📍 Distance | Routes, rivières, lignes électriques, casernes |
| 🌡️ Météo | Température max, vent, humidité, précipitations |
| 📈 Anomalies | Écart standardisé à la moyenne sur 30 ans |
| 🌿 Végétation | Classe de végétation dominante |
| 🏔️ Terrain | Élévation, pente, orientation |

**Target** : `ignition` (1 = point d'ignition, 0 = non-ignition) — classes très déséquilibrées (~1:5.6).

---

## 🤖 Modèle

- **Algorithme** : XGBoost
- **Tuning** : `RandomizedSearchCV`
- **Métrique optimisée** : **PR-AUC** 

---

## 🚀 Installation & Utilisation

### Prérequis

- [uv](https://docs.astral.sh/uv/) installé
- Python 3.13+
- **macOS (Silicon M1/M2/M3) :** L'entraînement XGBoost nécessite la bibliothèque OpenMP.
  ```bash
  brew install libomp
  ```

### Installation

```bash
git clone https://github.com/hugzweek/projet_meppds.git
cd projet_meppds
uv sync
```

## 📊 Suivi des expériences (MLflow)

Ce projet utilise **MLflow** pour centraliser le suivi des entraînements, comparer les performances des modèles et stocker les artefacts (modèles et préprocesseurs).

### 🔑 Configuration & Secrets

Avant de lancer l'entraînement, vous devez configurer vos accès au serveur MLflow. Le projet utilise un fichier `.env` pour charger les secrets de manière sécurisée.

1. **Initialiser votre environnement** :
   ```bash
   cp .env.example .env
   ```

2. **Configurer les variables** :
Ouvrez le fichier .env et complétez les informations suivantes :
- MLFLOW_TRACKING_URI : L'URL du serveur MLflow (par défaut, celle du service partagé).
- MLFLOW_TRACKING_USERNAME : Votre identifiant (ou celui du service partagé).
- MLFLOW_TRACKING_PASSWORD : Votre jeton d'accès (ou celui du service partagé).

[!IMPORTANT]
Le fichier `.env.example` est pré-rempli avec les identifiants du **service partagé** du projet. Normalement, aucune modification n'est nécessaire pour contribuer au serveur commun. Le fichier `.env` final est ignoré par Git pour protéger vos accès.

### 🏋️ Entraînement
Pour lancer le pipeline d'entraînement et enregistrer les résultats :

    ```bash
    uv run --env-file .env python train.py
    ```

Options disponibles :
Le script supporte plusieurs arguments pour personnaliser l'exécution :

Mode de tracking (--mode) :

- shared (par défaut) : Enregistre sur l'instance MLflow commune de l'équipe.
- personal : Enregistre sur votre propre instance MLflow Onyxia (nécessite de mettre à jour l'URL dans votre .env).

Paramètres du modèle :

--experiment_name : Nom de l'expérience dans l'interface MLflow (par défaut : wildfire_ML).

--n_estimators : Nombre d'arbres pour le modèle XGBoost (par défaut : 300).

--cv : Nombre de folds pour la cross-validation (par défaut : 5).

Exemple de commande personnalisée :

```bash
uv run --env-file .env python train.py --mode personal --n_estimators 500 --cv 3
```

📈 Visualisation
Une fois l'entraînement terminé, vous pouvez consulter les scores (PR-AUC), comparer les versions et récupérer les modèles enregistrés (Model Registry) sur l'interface web :

👉 [Accéder au Serveur MLflow du projet](https://user-gb53-mlflow.user.lab.sspcloud.fr)

### Lancer l'API en local

```bash
uv run uvicorn app.api:app --reload
```

→ [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🐳 Docker

```bash
# Builder l'image
docker build -t wildfire-api .

# Lancer le container
docker run -p 8000:8000 wildfire-api
```

L'image est automatiquement buildée et publiée sur Docker Hub.

---

## ☁️ Déploiement

L'API est déployée sur le **SSP Cloud** (Kubernetes) :

- **URL publique** : [https://wildfire-api.user.lab.sspcloud.fr/docs](https://wildfire-api.user.lab.sspcloud.fr/docs)
- **Orchestration** : Kubernetes (Deployment + Service + Ingress)
- **GitOps** : ArgoCD surveille le dépôt [application-deployment](https://github.com/hugzweek/application-deployment) et redéploie automatiquement à chaque changement

---


## 📄 Licence

Ce projet est sous licence **MIT** — voir le fichier [LICENSE](LICENSE).
