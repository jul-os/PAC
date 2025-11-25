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

from numbers import Integral, Real
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import issparse

df = pd.read_csv("/home/julia/Рабочий стол/code/PAC/titanic_ML/titanic_prepared.csv")
X = df.drop(["label"], axis=1)
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, shuffle=True
)


def _get_n_samples_bootstrap(n_samples, max_samples):
    """
    Get the number of samples in a bootstrap sample.
    """
    if isinstance(max_samples, Integral):
        if max_samples > n_samples:
            msg = "`max_samples` must be <= n_samples={} but got value {}"
            raise ValueError(msg.format(n_samples, max_samples))
        return max_samples

    if isinstance(max_samples, Real):
        return max(round(n_samples * max_samples), 1)


class MyRandomForest:

    def __init__(
        self,
        N_samples,
        M_dimension_feature_space,
        m_selected_features,
        _n_samples_bootstrap,
        estimator=DecisionTreeClassifier,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        n_estimators=100,  # количествао деревьев в лесу
        min_samples_leaf=1,
        random_state=42,
        max_samples=None,
    ):
        pass

    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y).
        """
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()
        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples=X.shape[0], max_samples=self.max_samples
        )
        # 436

        self._n_samples_bootstrap = n_samples_bootstrap
