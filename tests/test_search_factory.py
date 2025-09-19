import pytest
import numpy as np

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from AutomationML.search import SearchFactory, run_optuna_manual, gen_objective
from AutomationML.errors import UnknownSearchMethodError, OptionalDependencyError


def _toy_cls():
    X, y = make_classification(n_samples=300, n_features=10, n_informative=5, random_state=0)
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)


def _toy_reg():
    X, y = make_regression(n_samples=300, n_features=8, noise=0.2, random_state=0)
    return train_test_split(X, y, test_size=0.25, random_state=1)


def test_grid_search_logistic_regression_runs_and_has_best():
    X_train, X_valid, y_train, y_valid = _toy_cls()
    factory = SearchFactory(random_state=42)
    search = factory.create(
        method="grid",
        estimator_path="sklearn.linear_model.LogisticRegression",
        estimator_kwargs={"max_iter": 5000},
        param_space={
            "C": ("float", 0.1, 1.0, {"step": 0.45}),
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    assert hasattr(search, "best_params_")
    assert hasattr(search, "best_estimator_")
    assert search.best_estimator_.score(X_valid, y_valid) >= 0.7


def test_randomized_search_with_step_no_scipy_needed():
    # используем step → будет дискретный список (scipy не требуется)
    X_train, X_valid, y_train, y_valid = _toy_cls()
    factory = SearchFactory(random_state=0)
    search = factory.create(
        method="random",
        estimator_path="sklearn.ensemble.RandomForestClassifier",
        param_space={
            "n_estimators": ("int", 50, 150, {"step": 50}),
            "max_depth": ("int", 3, 9, {"step": 3}),
            "criterion": ["gini", "entropy"],
        },
        cv=3,
        scoring="accuracy",
        n_iter=5,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    assert hasattr(search, "best_params_")
    assert search.best_estimator_.score(X_valid, y_valid) >= 0.7


def test_optuna_manual_or_searchcv_both_paths_work(monkeypatch):
    optuna = pytest.importorskip("optuna")

    X_train, X_valid, y_train, y_valid = _toy_cls()
    factory = SearchFactory(random_state=42)

    res = factory.create(
        method="optuna",
        estimator_path="sklearn.svm.SVC",
        param_space={
            "C": ("float", 1e-3, 1e1, {"log": True}),
            "kernel": ["linear", "rbf"],
            "gamma": ("float", 1e-4, 1.0, {"log": True}),
        },
        cv=3,
        scoring="accuracy",
        n_trials=10,
        n_jobs=-1,
    )

    if hasattr(res, "fit"):
        res.fit(X_train, y_train)
        best = res.best_estimator_
        assert best.score(X_valid, y_valid) >= 0.7
    else:
        mode, cfg = res
        assert mode == "optuna-manual"
        study = run_optuna_manual(
            estimator_class=cfg["estimator_class"],
            grid=cfg["param_space"],
            X_train=X_train,
            y_train=y_train,
            scoring=cfg["scoring"],
            cv=cfg["cv"],
            n_trials=cfg["n_trials"],
            study_direction=cfg.get("study_direction", "maximize"),
        )
        from sklearn.svm import SVC
        model = SVC(**study.best_trial.params).fit(X_train, y_train)
        assert model.score(X_valid, y_valid) >= 0.7


def test_gen_objective_standalone():
    optuna = pytest.importorskip("optuna")
    from sklearn.linear_model import Ridge

    X_train, X_valid, y_train, y_valid = _toy_reg()
    objective = gen_objective(
        estimator_class=Ridge,
        grid={"alpha": ("float", 1e-6, 1e-1, {"log": True})},
        X_train=X_train,
        y_train=y_train,
        scoring="neg_mean_squared_error",
        cv=3,
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=8)
    assert "alpha" in study.best_trial.params


def test_unknown_method_raises():
    X_train, X_valid, y_train, y_valid = _toy_cls()
    factory = SearchFactory()
    with pytest.raises(UnknownSearchMethodError):
        factory.create(method="abrakadabra", estimator_path="sklearn.svm.SVC", param_space={})


def test_optuna_missing_raises(monkeypatch):
    import AutomationML.search as S
    monkeypatch.setattr(S, "_HAS_OPTUNA", False, raising=True)
    from AutomationML.search import SearchFactory
    with pytest.raises(OptionalDependencyError):
        SearchFactory().create(method="optuna", estimator_path="sklearn.svm.SVC", param_space={})


def test_optuna_manual_when_searchcv_absent(monkeypatch):
    import AutomationML.search as S
    optuna = pytest.importorskip("optuna")
    monkeypatch.setattr(S, "_HAS_OPTUNA", True, raising=True)
    monkeypatch.setattr(S, "_OptunaSearchCV", None, raising=True)
    from AutomationML.search import SearchFactory
    res = SearchFactory().create(method="optuna", estimator_path="sklearn.svm.SVC", param_space={}, n_trials=2)
    assert isinstance(res, tuple) and res[0] == "optuna-manual"

