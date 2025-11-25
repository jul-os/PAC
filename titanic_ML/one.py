"""
## Лабораторная 9.1

1. Загрузить файл, разделить его на train и test. Для test взять 10% случайно выбранных
строк таблицы.
2. Обучить модели: Decision Tree, XGBoost, Logistic Regression из библиотек sklearn и xgboost.
Обучить модели предсказывать столбец label по остальным столбцам таблицы.
3. Наладить замер Accuracy - доли верно угаданных ответов. -- уже потоу что используется accuracy_score
4. Точности всех моделей не должны быть ниже 85%
5. С помощью Decision Tree выбрать 2 самых важных признака и проверить точность модели,
обученной только на них.
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("/home/julia/Рабочий стол/code/PAC/titanic_ML/titanic_prepared.csv")
X = df.drop(["label"], axis=1)
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, shuffle=True
)
results = {}

# Decision Tree
dt_param_grid = {
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"],
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

dt_grid.fit(X_train, y_train)
best_tree = dt_grid.best_estimator_
tree_test_score = accuracy_score(y_test, best_tree.predict(X_test))

results["Decision Tree"] = {
    "best_params": dt_grid.best_params_,
    "test_score": tree_test_score,
}

# XGBoost
xgb_param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 6, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 0.9, 1.0],
    "gamma": [0, 0.1],
    "reg_alpha": [0, 0.1],
    "reg_lambda": [1, 1.5],
}

xgb_grid = GridSearchCV(XGBClassifier(random_state=42), xgb_param_grid, cv=5, n_jobs=-1)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_
xgb_test_score = accuracy_score(y_test, best_xgb.predict(X_test))

results["XGBoost"] = {
    "best_params": xgb_grid.best_params_,
    "test_score": xgb_test_score,
}

# Logistic Regression
lr_param_grid = {
    "C": [1e-4, 0.001, 0.01, 0.1, 10, 100, 1000, 10000],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"],
}

lr_grid = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=1000), lr_param_grid, cv=5, n_jobs=-1
)
lr_grid.fit(X_train, y_train)
best_lr = lr_grid.best_estimator_
lr_test_score = accuracy_score(y_test, best_lr.predict(X_test))

results["Logistic Regression"] = {
    "best_params": lr_grid.best_params_,
    "test_score": lr_test_score,
}


for model_name, result in results.items():
    print(f"\n{model_name}:")
    print(f"  Лучшие параметры: {result['best_params']}")
    print(f"  Точность на тесте: {result['test_score']:.4f}")

best_model_name = max(results.keys(), key=lambda x: results[x]["test_score"])
best_score = results[best_model_name]["test_score"]
print(f"ЛУЧШАЯ МОДЕЛЬ: {best_model_name} с точностью {best_score:.4f}")

feature_importances = best_tree.feature_importances_
feature_names = X.columns

# создаем табличку с признаками
feat_imporat_df = pd.DataFrame(
    {"feature": feature_names, "importance": feature_importances}
).sort_values("importance", ascending=False)
print(feat_imporat_df)
top_2_features = feat_imporat_df["feature"].head(2).tolist()
print(top_2_features)


def evaluate_models_on_features(selected_features, feature_set_name):

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    results_selected = {}
    best_dt_selected = XGBClassifier(**dt_grid.best_params_, random_state=42)
    best_dt_selected.fit(X_train_selected, y_train)
    dt_test_score = accuracy_score(y_test, best_dt_selected.predict(X_test_selected))
    results_selected["Decision Tree"] = {
        "test_score": dt_test_score,
    }
    best_xgb_selected = XGBClassifier(**xgb_grid.best_params_, random_state=42)
    best_xgb_selected.fit(X_train_selected, y_train)
    xgb_test_score = accuracy_score(y_test, best_xgb_selected.predict(X_test_selected))
    results_selected["XGBoost"] = {
        "test_score": xgb_test_score,
    }
    best_lr_selected = LogisticRegression(
        **lr_grid.best_params_, random_state=42, max_iter=1000
    )
    best_lr_selected.fit(X_train_selected, y_train)
    lr_test_score = accuracy_score(y_test, best_lr_selected.predict(X_test_selected))
    results_selected["Logistic Regression"] = {
        "test_score": lr_test_score,
    }
    return results_selected


results_2_features = evaluate_models_on_features(
    top_2_features, "2 самых важных признака"
)

print(f"\n{'Модель':<20} {'Все признаки':<12} {'2 признака':<12}")
for model_name in results.keys():
    original_score = results[model_name]["test_score"]
    score_2_features = results_2_features[model_name]["test_score"]

    print(f"{model_name:<20} {original_score:<12.4f}{score_2_features:<12.4f}")
