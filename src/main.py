import argparse
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .preprocessing import load_data, prepare_features, remove_outliers, split_and_scale
from .models import get_model, needs_scaling


def parse_args():
    parser = argparse.ArgumentParser(description="ML Pipeline: Atomization Energy")
    parser.add_argument("--model", type=str, default="GradientBoosting",
                        choices=["LinearRegression", "Ridge", "Lasso",
                                 "RandomForest", "GradientBoosting", "SVR", "KNN"])
    parser.add_argument("--data", type=str, default="data/roboBohr.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Создаём каталог с временной меткой
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output) / f"{args.model}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(exist_ok=True)
    print(f"[INFO] Каталог запуска: {run_dir}")

    # Пайплайн
    df = load_data(args.data)
    X, y = prepare_features(df)
    X, y = remove_outliers(X, y)
    X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler = split_and_scale(
        X, y, random_state=args.seed
    )

    # Обучение
    model = get_model(args.model, args.seed)
    X_tr = X_train_s if needs_scaling(args.model) else X_train
    X_te = X_test_s if needs_scaling(args.model) else X_test

    print(f"[INFO] Обучение модели: {args.model}")
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    # Метрики
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }
    print(f"[INFO] Метрики: {metrics}")

    # Сохранение
    joblib.dump(model, run_dir / "models" / f"{args.model}.pkl")
    joblib.dump(scaler, run_dir / "models" / "scaler.pkl")
    pd.DataFrame([metrics]).to_csv(run_dir / "metrics.csv", index=False)

    print(f"[OK] Результаты сохранены в {run_dir}")


if __name__ == "__main__":
    main()
