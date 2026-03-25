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

### Installation

```bash
git clone https://github.com/hugzweek/projet_meppds.git
cd projet_meppds
uv sync
```

### Entraînement

```bash
uv run train.py
```

Options disponibles :

```bash
uv run train.py --experiment_name wildfire_ML --n_estimators 300 --cv 5
```

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
