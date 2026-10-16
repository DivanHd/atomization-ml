from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


def get_model(name: str, seed: int = 42):
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, max_iter=10000),
        "Lasso": Lasso(alpha=0.01, max_iter=10000, random_state=seed),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=15,
            random_state=seed, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=5,
            random_state=seed
        ),
        "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
        "KNN": KNeighborsRegressor(n_neighbors=5),
    }
    if name not in models:
        raise ValueError(f"Модель '{name}' не найдена")
    return models[name]


def needs_scaling(model_name: str) -> bool:
    return model_name in ["LinearRegression", "Ridge", "Lasso", "SVR", "KNN"]
