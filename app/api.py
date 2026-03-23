"""A simple API to expose our trained XGBoost model for Wildfire Ignition prediction."""

from fastapi import FastAPI
import skops.io as sio
import pandas as pd

unknown_types = sio.get_untrusted_types(file="model.skops")
model = sio.load("model.skops", trusted=unknown_types)

app = FastAPI(
    title="Wildfire Ignition Prediction API",
    description="<b>Application de prédiction de point d'ignition de feux de forêt</b> 🔥"
    + '<br><br><img src="https://minio.lab.sspcloud.fr/wildfire-mlops/static/wildfire.png" width="300">',
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
    distance_fire_stations: float = 1500.0,
    distance_rivers: float = 800.0,
    distance_roads: float = 200.0,
    distance_powerlines: float = 600.0,
    aspect: float = 180.0,
    elevation: float = 400.0,
    pop_dens: float = 50.0,
    slope: float = 5.0,
    anom_max_temp: float = 1.5,
    anom_max_wind_vel: float = 0.3,
    anom_avg_temp: float = 0.8,
    anom_avg_rel_hum: float = -2.0,
    anom_avg_soil: float = -0.1,
    anom_sum_prec: float = -5.0,
    max_temp: float = 32.0,
    max_wind_vel: float = 45.0,
    avg_wind_angle: float = 180.0,
    avg_rel_hum: float = 30.0,
    avg_soil: float = 0.12,
    sum_prec: float = 0.0,
    vegetation_class: str = "forest",
) -> str:
    """
    Prédit si une zone va s'enflammer à partir des données météo et environnementales.

    - **vegetation_class** : forest, cropland, shrubland, herbaceous_vegetation, urban, water, wetland
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

    return "🔥 Ignition probable" if prediction == 1 else "✅ Pas d'ignition"