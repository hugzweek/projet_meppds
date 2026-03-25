"""A simple API to expose our trained XGBoost model for Wildfire Ignition prediction."""

from fastapi import FastAPI, Query
import skops.io as sio
import pandas as pd
import urllib.request

urllib.request.urlretrieve(
    "https://minio.lab.sspcloud.fr/hugoseumen/wildfire-mlops/models/model.skops",
    "model.skops"
)

unknown_types = sio.get_untrusted_types(file="model.skops")
model = sio.load("model.skops", trusted=unknown_types)

app = FastAPI(
    title="🔥 Wildfire Ignition Point Prediction",
    description="""
<b>Prédiction de points d'ignition de feux de forêt</b>

Cette API prédit si un point géographique est susceptible d'être un **point d'ignition** 
d'un feu de forêt, à partir de données météorologiques et environnementales.

<br>

## Features utilisées
- 📍 **Distance** aux routes, rivières, lignes électriques, casernes
- 🌡️ **Météo** : température, vent, humidité, précipitations
- 🌿 **Végétation** : classe de végétation dominante
- 🏔️ **Terrain** : élévation, pente, orientation


<br><br>
<img src="https://minio.lab.sspcloud.fr/hugoseumen/wildfire-mlops/static/wildfire.png" width="400">
""",
    version="0.0.1",
)

@app.get("/", tags=["Welcome"])
def show_welcome_page():
    """Show welcome page with model name and version."""
    return {
        "Message": "API de prédiction d'ignition de feux de forêt",
        "Model_name": "Wildfire XGBoost",
        "Model_version": "0.0.1",
    }


@app.get("/predict", tags=["Predict"])
async def predict(
    distance_fire_stations: float = Query(default=1500.0, description="Distance à la caserne la plus proche (m)"),
    distance_rivers: float = Query(default=800.0, description="Distance à la rivière la plus proche (m)"),
    distance_roads: float = Query(default=200.0, description="Distance à la route la plus proche (m)"),
    distance_powerlines: float = Query(default=600.0, description="Distance à la ligne électrique la plus proche (m)"),
    aspect: float = Query(default=180.0, description="Orientation de la pente (degrés)"),
    elevation: float = Query(default=400.0, description="Altitude (m)"),
    pop_dens: float = Query(default=50.0, description="Densité de population (personnes/km²)"),
    slope: float = Query(default=5.0, description="Pente (degrés)"),
    anom_max_temp: float = Query(default=1.5, description="Anomalie de température max (écart à la moyenne 30 ans)"),
    anom_max_wind_vel: float = Query(default=0.3, description="Anomalie de vitesse du vent max"),
    anom_avg_temp: float = Query(default=0.8, description="Anomalie de température moyenne"),
    anom_avg_rel_hum: float = Query(default=-2.0, description="Anomalie d'humidité relative moyenne"),
    anom_avg_soil: float = Query(default=-0.1, description="Anomalie d'humidité du sol moyenne"),
    anom_sum_prec: float = Query(default=-5.0, description="Anomalie de précipitations cumulées"),
    max_temp: float = Query(default=32.0, description="Température maximale du jour (°C)"),
    max_wind_vel: float = Query(default=45.0, description="Vitesse maximale du vent (m/s)"),
    avg_wind_angle: float = Query(default=180.0, description="Angle moyen du vent (degrés)"),
    avg_rel_hum: float = Query(default=30.0, description="Humidité relative moyenne (%)"),
    avg_soil: float = Query(default=0.12, description="Humidité du sol moyenne (m³/m³)"),
    sum_prec: float = Query(default=0.0, description="Précipitations cumulées du jour (mm)"),
    vegetation_class: str = Query(
        default="forest",
        description="Classe de végétation dominante : forest, cropland, shrubland, herbaceous_vegetation, urban, water, wetland"
    ),
) -> str:
    """
    Prédit si un point géographique est un **point d'ignition** de feu de forêt.
 
    Retourne :
    - 🔥 **Ignition probable** si le modèle prédit une ignition
    - ✅ **Pas d'ignition** sinon
    """
 
    df = pd.DataFrame({
        "distance_fire_stations": [distance_fire_stations],
        "distance_rivers": [distance_rivers],
        "distance_roads": [distance_roads],
        "distance_powerlines": [distance_powerlines],
        "aspect": [aspect],
        "elevation": [elevation],
        "pop_dens": [pop_dens],
        "slope": [slope],
        "anom_max_temp": [anom_max_temp],
        "anom_max_wind_vel": [anom_max_wind_vel],
        "anom_avg_temp": [anom_avg_temp],
        "anom_avg_rel_hum": [anom_avg_rel_hum],
        "anom_avg_soil": [anom_avg_soil],
        "anom_sum_prec": [anom_sum_prec],
        "max_temp": [max_temp],
        "max_wind_vel": [max_wind_vel],
        "avg_wind_angle": [avg_wind_angle],
        "avg_rel_hum": [avg_rel_hum],
        "avg_soil": [avg_soil],
        "sum_prec": [sum_prec],
        "vegetation_class": [vegetation_class],
    })
 
    proba = model.predict_proba(df)[0][1]
    prediction = int(proba >= 0.5)
 
    return f"🔥 Ignition probable (probabilité : {proba:.1%})" if prediction == 1 else f"✅ Pas d'ignition (probabilité : {proba:.1%})"