import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from feature_engine.encoding import CountFrequencyEncoder
from sklearn import set_config
set_config(transform_output = "pandas")

log = logging.getLogger(__name__)

VEGETATION_COLS = [
    "cropland", "herbaceous_vegetation", "moss_lichen", "shrubland",
    "sprarse_vegetation", "urban", "water", "wetland", "forest",
]
FOREST_TYPE_COLS = [
    "forest_deciduous_needle", "forest_evergreen_broad",
    "forest_deciduous_broad", "forest_evergreen_needle",
    "forest_mixed", "forest_unknown",
]
COLS_TO_DROP = ["avg_temp", "max_max_temp", "Year", "yearly_avg_temp"]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Drop rows with the 5 NAs patern
    five_na_col = data.isna().sum().sort_values(ascending=False).iloc[2:20].index
    data = data.loc[~(data[five_na_col].isna().any(axis=1))]
    log.info(f"After NA removal: {data.shape}")

    # Fix vegetation_class typos
    data.loc[data["vegetation_class"] == "$herb$aceous_vegetation", "vegetation_class"] = "herbaceous_vegetation"
    data.loc[data["vegetation_class"] == "Forestt", "vegetation_class"] = "forest"

    # Fill remaining vegetation_class NAs
    data.loc[data["vegetation_class"].isna(), "vegetation_class"] = "cropland"

    # Fill yearly_avg_temp per year
    for year in data["Year"].unique():
        mask = data["Year"] == year
        year_avg = data.loc[mask, "yearly_avg_temp"].mean()
        data.loc[mask, "yearly_avg_temp"] = data.loc[mask, "yearly_avg_temp"].fillna(year_avg)

    # Set date as index
    data.set_index("Date", inplace=True)

    # Drop vegetation detail columns + low-signal/correlated columns
    data = data.drop(columns=VEGETATION_COLS + FOREST_TYPE_COLS + COLS_TO_DROP, errors="ignore")

    log.info(f"After preprocessing: {data.shape}")
    return data


def build_features(
    data: pd.DataFrame,
    target_col: str,
    encoding: str = "onehot",
    test_size: float = 0.3,
    random_state: int = 1,
):
    X = data.drop(columns=[target_col])
    y = data[target_col]

    float_cols = X.select_dtypes(include="float").columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    cat_transformer = (
        OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        if encoding == "onehot"
        else CountFrequencyEncoder(encoding_method="frequency")
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), float_cols),
            ("cat", cat_transformer, cat_cols),
        ]
    )

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = np.array(X.columns.tolist())


    log.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, preprocessor, feature_names