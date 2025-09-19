import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from AutomationML.search import SearchFactory, gen_objective, run_optuna_manual


def _toy_cls():
    X, y = make_classification(n_samples=220, n_features=8, n_informative=4, random_state=0)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)


def _toy_reg():
    X, y = make_regression(n_samples=220, n_features=6, noise=0.2, random_state=0)
    return train_test_split(X, y, test_size=0.3, random_state=1)


def test_gen_objective_callable_and_int_float_steps_and_less_is_better():
    optuna = pytest.importorskip("optuna")
    from sklearn.linear_model import Ridge

    X_train, X_valid, y_train, y_valid = _toy_reg()

    # callable в пространстве + int/float со step
    def alpha_spec(trial):
        # проверяем ветку callable(trial)
        return trial.suggest_float("alpha", 1e-4, 1e-2, log=True)

    grid = {
        "alpha": alpha_spec,                      # callable
        "max_iter": ("int", 1000, 2000, {"step": 500}),   # int со step
        "tol": ("float", 1e-5, 1e-3, {"step": 5e-5}),     # float со step
    }

    # greater_is_better=False покроет ветку инвертирования метрики
    objective = gen_objective(
        estimator_class=Ridge,
        grid=grid,
        X_train=X_train,
        y_train=y_train,
        scoring="r2",
        cv=2,
        greater_is_better=False,
    )
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=3)
    assert isinstance(study.best_trial.value, float)
    assert "alpha" in study.best_trial.params
    assert "max_iter" in study.best_trial.params
    assert "tol" in study.best_trial.params


def test_run_optuna_manual_with_sampler_pruner_callbacks_and_timeout():
    optuna = pytest.importorskip("optuna")
    from sklearn.svm import SVC

    X_train, X_valid, y_train, y_valid = _toy_cls()

    calls = {"cnt": 0}

    def cb(study, trial):
        calls["cnt"] += 1

    study = run_optuna_manual(
        estimator_class=SVC,
        grid={
            "C": ("float", 1e-3, 1e1, {"log": True}),
            "kernel": ["linear", "rbf"],
            "gamma": ("float", 1e-4, 1.0, {"log": True}),
        },
        X_train=X_train,
        y_train=y_train,
        scoring="accuracy",
        cv=2,
        n_trials=4,
        study_direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        timeout=10,
        callbacks=[cb],
    )
    assert len(study.trials) >= 1
    assert calls["cnt"] >= 1  # хотя бы один колбэк был вызван


def test_factory_optuna_path_with_integration(monkeypatch):
    """
    Покрываем ветку, когда доступен optuna.integration.OptunaSearchCV.
    Подменяем её на простую заглушку с совместимым API.
    """
    from AutomationML import search as S

    class DummyOptunaSearchCV:
        def __init__(self, estimator, param_distributions, cv, scoring, n_trials,
                     n_jobs, refit, random_state):
            self.estimator = estimator
            self.param_distributions = param_distributions
            self.cv = cv
            self.scoring = scoring
            self.n_trials = n_trials
            self.n_jobs = n_jobs
            self.refit = refit
            self.random_state = random_state
            self.best_estimator_ = None

        def fit(self, X, y):
            # просто обучаем переданный estimator один раз
            self.best_estimator_ = self.estimator.fit(X, y)
            return self

    optuna = pytest.importorskip("optuna")

    monkeypatch.setattr(S, "_HAS_OPTUNA", True, raising=True)
    monkeypatch.setattr(S, "_OptunaSearchCV", DummyOptunaSearchCV, raising=True)

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
        cv=2,
        scoring="accuracy",
        n_trials=5,
        n_jobs=-1,
    )
    # уверены, что вернулся совместимый объект (а не кортеж ручного режима)
    assert hasattr(res, "fit")
    res.fit(X_train, y_train)
    assert hasattr(res, "best_estimator_")
    assert res.best_estimator_.score(X_valid, y_valid) >= 0.6


import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from AutomationML.search import SearchFactory


def _toy():
    X, y = make_classification(n_samples=160, n_features=10, n_informative=5, random_state=0)
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)


def test_resolve_estimator_uses_instance_and_ignores_path():
    # передаём готовый экземпляр → фабрика должна его использовать
    est = RandomForestClassifier(n_estimators=10, random_state=123)
    f = SearchFactory(random_state=999)  # не должен перезаписать random_state готового экземпляра
    # используем публичный API create; если передан estimator — path/kwargs игнорируются
    s = f.create(
        method="grid",
        estimator=est,
        estimator_path="sklearn.ensemble.RandomForestClassifier",
        estimator_kwargs={"n_estimators": 999},  # должен быть проигнорирован
        param_space={"n_estimators": ("int", 10, 10, {"step": 1})},
        cv=2,
    )
    # внутри GridSearchCV должен быть именно наш экземпляр (по крайней мере по важным признакам)
    assert s.estimator is est
    assert s.estimator.random_state == 123


def test_bad_estimator_kwargs_raises_value_error():
    f = SearchFactory()
    with pytest.raises(ValueError) as ei:
        f.create(
            method="grid",
            estimator_path="sklearn.linear_model.LogisticRegression",
            estimator_kwargs={"n_estimator": 10},  # опечатка
            param_space={"C": ("float", 1.0, 1.0, {"step": 1.0})},
            cv=2,
        )
    assert "Недопустимые параметры" in str(ei.value)


def test_grid_multimetric_refit_string_path():
    # мульти-метрики + refit='acc' покрывают ветки передачи refit
    X_train, X_valid, y_train, y_valid = _toy()
    f = SearchFactory()
    s = f.create(
        method="grid",
        estimator_path="sklearn.linear_model.LogisticRegression",
        estimator_kwargs={"max_iter": 2000},
        param_space={"C": ("float", 1.0, 1.0, {"step": 1.0})},
        cv=2,
        scoring={"acc": "accuracy", "f1": "f1"},
        refit="acc",
        n_jobs=1,
    )
    s.fit(X_train, y_train)
    assert "mean_test_acc" in s.cv_results_
    assert "mean_test_f1" in s.cv_results_