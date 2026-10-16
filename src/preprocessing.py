import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    print(f"[INFO] Данные загружены: {df.shape}")
    return df


def prepare_features(df: pd.DataFrame):
    target_col = "Eat"
    drop_cols = [target_col, "pubchem_id", "Unnamed: 0", "xyz"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    return X, y


def remove_outliers(X: pd.DataFrame, y: pd.Series):
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = ~((y < lower) | (y > upper))
    print(f"[INFO] После удаления выбросов: {X[mask].shape}")
    return X[mask], y[mask]


def split_and_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler
