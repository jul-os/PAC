"""
## Лабораторная 9.2
1. Лабораторная 9.1 пп.1-5
2. Реализовать случайный лес в виде класса MyRandomForest.
В реализации разрешается использовать DecisionTreeClassifier из библиотеки sklearn.
 Класс должен иметь методы fit и predict по аналогии с остальными классами библиотеки sklearn.
    Алгоритм построения Случайного леса изложен на [Википедии](https://ru.wikipedia.org/wiki/Random_forest)
    Необходимо обратить внимание что при построения леса используются не все доступные признаки
    для каждого узла дерева. А так же что в sklearn это регулируется параметрами DecisionTreeClassifier.
3. Продемонстрировать, что точность леса выше чем точность одного решающего дерева.

"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


class MyRandomForest:
    def __init__(self, n_estimators=100, max_features="sqrt", random_state=None):
        """
        Конструктор случайного леса

        Parameters:
        -----------
        n_estimators : int, default=100
            Количество деревьев в лесу
        max_features : str or int, default='sqrt'
            Количество признаков для рассмотрения в каждом разбиении:
            - 'sqrt': sqrt(n_features)
            - 'log2': log2(n_features)
            - int: конкретное число признаков
        random_state : int, default=None
            Seed для воспроизводимости результатов
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []
        self.n_features_ = None
        self.n_classes_ = None

    def _get_max_features(self, n_features):
        """Определяет количество признаков для каждого разбиения"""
        if self.max_features == "sqrt":
            return int(np.sqrt(n_features))
        elif self.max_features == "log2":
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        else:
            raise ValueError("max_features должен быть 'sqrt', 'log2' или int")

    def fit(self, X, y):
        """
        Обучение случайного леса

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Матрица признаков
        y : array-like of shape (n_samples,)
            Вектор целевых переменных
        """
        X = np.array(X)
        y = np.array(y)

        self.n_features_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))
        n_samples = X.shape[0]

        # Определяем количество признаков для разбиения
        m = self._get_max_features(self.n_features_)

        # Инициализируем генератор случайных чисел
        rng = np.random.RandomState(self.random_state)

        self.trees = []
        self.feature_indices = []

        for i in range(self.n_estimators):
            # Генерируем бутстрап-подвыборку
            X_boot, y_boot = resample(
                X,
                y,
                random_state=(
                    self.random_state + i if self.random_state is not None else None
                ),
            )

            # Случайно выбираем признаки для этого дерева
            feature_idx = rng.choice(self.n_features_, m, replace=False)
            X_boot_subset = X_boot[:, feature_idx]

            # Создаем и обучаем дерево
            tree = DecisionTreeClassifier(
                max_features=None,  # Мы уже отобрали признаки
                random_state=(
                    self.random_state + i if self.random_state is not None else None
                ),
            )
            tree.fit(X_boot_subset, y_boot)

            # Сохраняем дерево и индексы признаков
            self.trees.append(tree)
            self.feature_indices.append(feature_idx)

        return self

    def predict(self, X):
        """
        Предсказание классов для входных данных

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Матрица признаков

        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Предсказанные классы, те список клкассов где для каждого дерева наиболее встречающийся класс
        """
        X = np.array(X)
        n_samples = X.shape[0]

        # будем собирать предсказания всех деревьев
        all_predictions = np.zeros(
            (self.n_estimators, n_samples)
        )  # матрица размера количество_деревьев * количество_образцов

        for i, (tree, feature_idx) in enumerate(zip(self.trees, self.feature_indices)):
            X_subset = X[:, feature_idx]
            all_predictions[i] = tree.predict(
                X_subset
            )  # используется готовое дерево из sklearn

        # выбираем наиболее частый класс для каждого образца
        y_pred = np.zeros(n_samples, dtype=int)
        for j in range(n_samples):
            unique, counts = np.unique(all_predictions[:, j], return_counts=True)
            y_pred[j] = unique[np.argmax(counts)]

        return y_pred

    def predict_proba(self, X):
        """
        Предсказание вероятностей классов

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Матрица признаков

        Returns:
        --------
        proba : array of shape (n_samples, n_classes)
            Вероятности классов
        """
        X = np.array(X)
        n_samples = X.shape[0]

        # Собираем предсказания всех деревьев
        all_predictions = np.zeros((self.n_estimators, n_samples))

        for i, (tree, feature_idx) in enumerate(zip(self.trees, self.feature_indices)):
            X_subset = X[:, feature_idx]
            all_predictions[i] = tree.predict(X_subset)

        # Вычисляем вероятности как долю деревьев, проголосовавших за каждый класс
        proba = np.zeros((n_samples, self.n_classes_))
        for j in range(n_samples):
            for k in range(self.n_classes_):
                proba[j, k] = np.sum(all_predictions[:, j] == k) / self.n_estimators

        return proba

    def get_params(self, deep=True):
        """Получить параметры модели"""
        return {
            "n_estimators": self.n_estimators,
            "max_features": self.max_features,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        """Установить параметры модели"""
        for key, value in params.items():
            setattr(self, key, value)
        return self


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

# MyRandomForest
rf_param_grid = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_features": ["sqrt", "log2", 0.3, 0.5],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
}

rf_grid = GridSearchCV(
    MyRandomForest(random_state=42),
    rf_param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

rf_grid.fit(X_train.values, y_train.values)
best_rf = rf_grid.best_estimator_
rf_test_score = accuracy_score(y_test, best_rf.predict(X_test.values))

results["MyRandomForest"] = {
    "best_params": rf_grid.best_params_,
    "test_score": rf_test_score,
}


for model_name, result in results.items():
    print(f"\n{model_name}:")
    print(f"  Лучшие параметры: {result['best_params']}")
    print(f"  Точность на тесте: {result['test_score']:.4f}")

best_model_name = max(results.keys(), key=lambda x: results[x]["test_score"])
best_score = results[best_model_name]["test_score"]
print(f"ЛУЧШАЯ МОДЕЛЬ: {best_model_name} с точностью {best_score:.4f}")


print(f"Точность Decision Tree: {results['Decision Tree']['test_score']:.4f}")
print(f"Точность MyRandomForest: {results['MyRandomForest']['test_score']:.4f}")
