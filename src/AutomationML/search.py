# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Motynga Sergey/OMSTU
"""
Фабрика стратегий поиска гиперпараметров с единым DSL пространства параметров.

Назначение
----------
Единый интерфейс для создания стратегий поиска гиперпараметров:
— GridSearchCV
— RandomizedSearchCV
— Optuna (через OptunaSearchCV, либо ручной режим study.optimize)

Используется единый формат описания пространства гиперпараметров (DSL), пригодный
для всех методов: категориальные параметры (список значений), числовые диапазоны
("int"/"float" с границами и опциями step/log), а также условные параметры через
callable(trial) для Optuna.

Соглашения
----------
— Идентификаторы методов: "grid" | "gridsearchcv", "random" | "randomizedsearchcv",
  "optuna" | "optunasearchcv".
— Пространство параметров (param_space):
   * Категории: список/кортеж примитивов.
     {"criterion": ["gini", "entropy", "log_loss"]}
   * Диапазоны: ("int"|"float", low, high, {opts?}), где opts:
       step: int|float — шаг дискретизации (обязателен для Grid; для Randomized задаёт дискретизацию),
       log: bool — логарифмическая шкала (Randomized/Optuna; для int без step игнорируется).
     {"max_depth": ("int", 2, 50),
      "alpha": ("float", 1e-4, 1.0, {"log": True}),
      "n_estimators": ("int", 100, 1000, {"step": 50})}
   * Условные/произвольные параметры для Optuna: callable(trial) -> value.
— Для GridSearchCV диапазоны должны иметь step (иначе InvalidSearchSpaceError).
— Для RandomizedSearchCV без step применяются распределения scipy.stats; со step — дискретные списки.
— Optuna: при наличии optuna.integration.OptunaSearchCV возвращается sklearn-совместимый объект;
  иначе доступен ручной режим через run_optuna_manual(...).

Безопасность
------------
Не подставляйте пространства параметров из непроверённых источников (в них могут быть callable).
Импорты классов по строковым путям выполняются через безопасную обёртку.

Пример использования
--------------------
from AutomationML.search import SearchFactory, run_optuna_manual

factory = SearchFactory(random_state=42)

# GridSearchCV (обязателен step в числовых диапазонах)
grid = factory.create(
    method="grid",
    estimator_path="sklearn.linear_model.LogisticRegression",
    estimator_kwargs={"max_iter": 10_000},
    param_space={"C": ("float", 1e-3, 1e2, {"step": 1.0}),
                 "penalty": ["l2"], "solver": ["lbfgs"]},
    cv=5, scoring="accuracy", n_jobs=-1,
)
grid.fit(X, y)
print(grid.best_params_, grid.best_score_)

# RandomizedSearchCV (непрерывные диапазоны → распределения)
rnd = factory.create(
    method="random",
    estimator_path="sklearn.linear_model.Ridge",
    param_space={"alpha": ("float", 1e-6, 1.0, {"log": True})},
    cv=5, scoring="neg_mean_squared_error", n_iter=30, n_jobs=-1,
)
rnd.fit(X, y)

# Optuna: OptunaSearchCV (если доступен) или ручной режим
opt = factory.create(
    method="optuna",
    estimator_path="sklearn.svm.SVC",
    param_space={"C": ("float", 1e-3, 1e3, {"log": True}),
                 "kernel": ["linear", "rbf"],
                 "gamma": ("float", 1e-4, 1.0, {"log": True})},
    cv=5, scoring="accuracy", n_trials=40, n_jobs=-1,
)
if hasattr(opt, "fit"):
    opt.fit(X, y)
    best_model = opt.best_estimator_
else:
    mode, cfg = opt
    study = run_optuna_manual(
        estimator_class=cfg["estimator_class"],
        grid=cfg["param_space"],
        X_train=X, y_train=y,
        scoring=cfg["scoring"], cv=cfg["cv"],
        n_trials=cfg["n_trials"],
        study_direction=cfg.get("study_direction", "maximize"),
        sampler=cfg.get("sampler"), pruner=cfg.get("pruner"),
        fit_params=cfg.get("fit_params"), fixed_params=cfg.get("fixed_params"),
        timeout=cfg.get("timeout"), callbacks=cfg.get("callbacks"),
    )
    from sklearn.svm import SVC
    best_model = SVC(**study.best_trial.params).fit(X, y)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
from dataclasses import dataclass

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

from .utils import (
    import_from_path,
    normalize_name,
    validate_kwargs,
    maybe_inject_random_state,
    json_safe,
    is_primitive_sequence,
    space_to_grid,
    space_to_random_distributions,
)
from .errors import OptionalDependencyError  # дополнительные исключения импортируются точечно ниже

try:  # pragma: no cover
    import optuna as _optuna
    _HAS_OPTUNA = True
    # Новый дом для интеграций (Optuna 3.6+)
    try:
        from optuna_integration import OptunaSearchCV as _OptunaSearchCV  # noqa: F401
    except Exception:
        try:
            from optuna.integration import OptunaSearchCV as _OptunaSearchCV  # noqa: F401
        except Exception:
            _OptunaSearchCV = None
except Exception:  # pragma: no cover
    _optuna = None
    _OptunaSearchCV = None
    _HAS_OPTUNA = False


ParamSpace = Dict[str, Any]
ScoreArg = Union[str, Callable[[Any, Any], float]]


def _space_to_optuna_dists(space: Dict[str, Any]) -> Dict[str, "_optuna.distributions.BaseDistribution"]:
    """
    Преобразует DSL-пространство параметров в словарь распределений Optuna,
    совместимый с OptunaSearchCV (из пакета ``optuna-integration`` или старого
    ``optuna.integration``).

    Формат входного DSL
    -------------------
    Поддерживаются:
      1) Категории: последовательность примитивов (list/tuple)
         >>> {"criterion": ["gini", "entropy"]}
         → CategoricalDistribution(["gini", "entropy"])

      2) Диапазоны: ("int"|"float", low, high, {opts?})
         - "int": opts {"step": int?, "log": bool?}
           * если step отсутствует → подставляется 1 (избегаем step=None)
           → IntDistribution(low, high, step, log)
         - "float": opts {"step": float|None, "log": bool?}
           * step может быть None (непрерывное распределение)
           → FloatDistribution(low, high, step|None, log)

      3) Одиночное значение → категория из одного элемента
         >>> {"max_depth": 5} → CategoricalDistribution([5])

    Ограничения
    -----------
    - callable(trial) в DSL здесь не поддерживается (это только для ручного режима
      ``run_optuna_manual``) — при обнаружении выбрасывается InvalidSearchSpaceError.

    Параметры
    ---------
    space : dict[str, Any]
        Пространство гиперпараметров (категории/диапазоны/константы).

    Возвращает
    ----------
    dict[str, optuna.distributions.BaseDistribution]
        Карта имя_параметра → распределение Optuna.

    Исключения
    ----------
    InvalidSearchSpaceError
        Если встретился callable или неизвестный тип диапазона.
    """
    from .errors import InvalidSearchSpaceError

    d: Dict[str, "_optuna.distributions.BaseDistribution"] = {}

    for name, spec in (space or {}).items():
        # 1) callable — не поддерживается в интеграции
        if callable(spec):
            raise InvalidSearchSpaceError(
                f"Параметр {name!r}: callable не поддерживается в OptunaSearchCV. "
                f"Используйте ручной режим run_optuna_manual(...)."
            )

        # 2) Категории
        if is_primitive_sequence(spec):
            d[name] = _optuna.distributions.CategoricalDistribution(list(spec))
            continue

        # 3) Диапазоны: ("int"/"float", low, high, {opts?})
        if isinstance(spec, tuple) and len(spec) >= 3 and isinstance(spec[0], str):
            kind = spec[0].lower()
            low, high = spec[1], spec[2]
            opts = spec[3] if len(spec) >= 4 and isinstance(spec[3], dict) else {}

            if kind == "int":
                step = opts.get("step")
                log = bool(opts.get("log", False))
                if step is None:
                    step = 1  # безопасное значение по умолчанию, избегаем step=None
                d[name] = _optuna.distributions.IntDistribution(
                    low=int(low), high=int(high), step=int(step), log=log
                )
                continue

            if kind == "float":
                step = opts.get("step")  # может быть None — допустимо
                log = bool(opts.get("log", False))
                d[name] = _optuna.distributions.FloatDistribution(
                    low=float(low),
                    high=float(high),
                    step=float(step) if step is not None else None,
                    log=log,
                )
                continue

            raise InvalidSearchSpaceError(
                f"Параметр {name!r}: неизвестный тип диапазона {kind!r}."
            )

        # 4) Константа → категория из одного значения
        d[name] = _optuna.distributions.CategoricalDistribution([spec])

    return d


@dataclass
class OptunaConfig:
    """
    Конфигурация параметров Optuna.

    Параметры
    ---------
    n_trials : int | None, optional (default=50)
        Число итераций оптимизации в Optuna.
    study_direction : str, optional (default="maximize")
        Направление оптимизации: "maximize" или "minimize".
    sampler : Any, optional
        Самплер Optuna (например, optuna.samplers.TPESampler(...)).
    pruner : Any, optional
        Прунер Optuna (например, optuna.pruners.MedianPruner(...)).
    timeout : int | None, optional
        Лимит времени оптимизации в секундах.
    callbacks : Iterable[Callable] | None, optional
        Колбэки для study.optimize(...).
    fixed_params : dict[str, Any] | None, optional
        Фиксированные параметры конструктора модели.
    fit_params : dict[str, Any] | None, optional
        Дополнительные параметры для .fit(...). Начиная с scikit-learn>=1.4,
        cross_val_score не поддерживает fit_params — используйте явный цикл CV при необходимости.

    Возвращаемые значения
    ---------------------
    OptunaConfig
        Объект конфигурации для Optuna.

    Исключения
    ----------
    Нет.
    """
    n_trials: Optional[int] = 50
    study_direction: str = "maximize"
    sampler: Any = None
    pruner: Any = None
    timeout: Optional[int] = None
    callbacks: Optional[Iterable[Callable[..., Any]]] = None
    fixed_params: Optional[Dict[str, Any]] = None
    fit_params: Optional[Dict[str, Any]] = None


def gen_objective(
    estimator_class: Union[type, Callable[..., BaseEstimator]],
    grid: ParamSpace,
    X_train: Any,
    y_train: Any,
    *,
    scoring: ScoreArg | None = None,
    cv: Optional[Union[int, Iterable]] = 5,
    fit_params: Optional[Dict[str, Any]] = None,
    greater_is_better: bool = True,
    fixed_params: Optional[Dict[str, Any]] = None,
) -> Callable[[" _optuna.trial.Trial"], float]:
    """
    Генерирует функцию цели Optuna под произвольную sklearn-модель.

    Параметры
    ---------
    estimator_class : type | Callable[..., BaseEstimator]
        Класс оценщика или фабрика, создающая экземпляры модели по параметрам.
    grid : dict[str, Any]
        Пространство параметров в формате DSL (категории, диапазоны, callable).
    X_train : Any
        Обучающие признаки.
    y_train : Any
        Обучающие ответы/метки.
    scoring : str | callable | None, optional (default=None)
        Метрика для cross_val_score. Совместима со sklearn.
    cv : int | Iterable | None, optional (default=5)
        Разбиение для перекрёстной проверки.
    fit_params : dict[str, Any] | None, optional
        Дополнительные параметры для .fit(...). Начиная с scikit-learn>=1.4, cross_val_score
        не принимает fit_params; используйте явный цикл по split'ам, если это необходимо.
    greater_is_better : bool, optional (default=True)
        Если False — возвращается отрицательное значение метрики (для минимизации).
    fixed_params : dict[str, Any] | None, optional
        Фиксированные параметры конструктора модели, не участвующие в поиске.

    Возвращаемые значения
    ---------------------
    callable
        Функция objective(trial) → float для передачи в Optuna.

    Исключения
    ----------
    OptionalDependencyError
        Optuna не установлена.
    """
    if not _HAS_OPTUNA:
        raise OptionalDependencyError(
            "Для использования gen_objective требуется 'optuna'. Установите: pip install optuna"
        )

    fit_params = fit_params or {}
    fixed_params = fixed_params or {}

    def _suggest(trial: " _optuna.trial.Trial", name: str, spec: Any) -> Any:
        if callable(spec):
            return spec(trial)
        if is_primitive_sequence(spec):
            return trial.suggest_categorical(name, list(spec))
        if isinstance(spec, tuple) and len(spec) >= 3 and isinstance(spec[0], str):
            kind = spec[0].lower()
            low, high = spec[1], spec[2]
            opts = spec[3] if len(spec) >= 4 and isinstance(spec[3], dict) else {}
            if kind == "int":
                step = opts.get("step", None)
                log = bool(opts.get("log", False))
                if step is None:
                    return trial.suggest_int(name, int(low), int(high), log=log)
                else:
                    return trial.suggest_int(name, int(low), int(high), step=int(step), log=log)
            if kind == "float":
                step = opts.get("step", None)
                log = bool(opts.get("log", False))
                if step is None:
                    return trial.suggest_float(name, float(low), float(high), log=log)
                else:
                    return trial.suggest_float(name, float(low), float(high), step=float(step), log=log)
        return spec

    def objective(trial: " _optuna.trial.Trial") -> float:
        params = {p: _suggest(trial, p, spec) for p, spec in (grid or {}).items()}
        estimator = estimator_class(**fixed_params, **params)
        # В sklearn>=1.4 cross_val_score не принимает fit_params — не передаём их здесь.
        scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring=scoring)
        mean_score = float(scores.mean())
        return mean_score if greater_is_better else -mean_score

    return objective


class SearchFactory:
    """
    Фабрика стратегий поиска гиперпараметров для sklearn и Optuna.

    Параметры
    ---------
    random_state : int | None, optional
        Значение по умолчанию для random_state, которое будет подставлено в конструктор
        оценщика, если он его поддерживает и random_state не указан в estimator_kwargs.
    """

    def __init__(self, random_state: Optional[int] = None) -> None:
        self.random_state = random_state

    def _resolve_estimator(
        self,
        estimator: Optional[BaseEstimator],
        estimator_path: Optional[str],
        estimator_kwargs: Optional[Dict[str, Any]],
    ) -> Tuple[type, BaseEstimator]:
        """
        Разрешает конструктор оценщика и, при необходимости, создаёт экземпляр.

        Параметры
        ---------
        estimator : BaseEstimator | None
            Готовый экземпляр модели (если уже создан).
        estimator_path : str | None
            Полный путь до класса оценщика (например, "sklearn.svm.SVC").
        estimator_kwargs : dict[str, Any] | None
            Параметры для конструктора оценщика (используются, если estimator не передан).

        Возвращаемые значения
        ---------------------
        tuple[type, BaseEstimator]
            Пара (класс_оценщика, экземпляр_оценщика).

        Исключения
        ----------
        ValueError
            Если не указан ни estimator, ни estimator_path.
        OptionalDependencyError
            Если модуль оценщика не установлен.
        ImportError
            Если класс оценщика не найден.
        """
        if estimator is not None:
            return type(estimator), estimator
        if not estimator_path:
            raise ValueError("Не указан ни готовый 'estimator', ни 'estimator_path' для его импорта.")
        cls = import_from_path(estimator_path)
        kwargs = dict(estimator_kwargs or {})
        if self.random_state is not None:
            kwargs = maybe_inject_random_state(kwargs, cls, self.random_state)
        msg = validate_kwargs(cls, kwargs)
        if msg:
            raise ValueError(f"Параметры конструктора оценщика некорректны:\n{msg}")
        return cls, cls(**kwargs)

    def create(
        self,
        method: str,
        *,
        estimator: Optional[BaseEstimator] = None,
        estimator_path: Optional[str] = None,
        estimator_kwargs: Optional[Dict[str, Any]] = None,
        param_space: Optional[ParamSpace] = None,
        cv: Optional[Union[int, Iterable]] = 5,
        scoring: ScoreArg | None = None,
        n_jobs: Optional[int] = None,
        refit: Union[bool, str] = True,
        random_state: Optional[int] = None,
        n_iter: Optional[int] = None,
        n_trials: Optional[int] = None,
        sampler: Any = None,
        pruner: Any = None,
        study_direction: str = "maximize",
        timeout: Optional[int] = None,
        callbacks: Optional[Iterable[Callable[..., Any]]] = None,
    ) -> Any:
        """
        Создаёт стратегию поиска гиперпараметров по ключу метода.

        Параметры
        ---------
        method : str
            Идентификатор метода: "grid" | "gridsearchcv" | "random" | "randomizedsearchcv" | "optuna" | "optunasearchcv".
        estimator : BaseEstimator | None, optional
            Готовый экземпляр оценщика. Если не указан, используется estimator_path + estimator_kwargs.
        estimator_path : str | None, optional
            Полный путь до класса оценщика, например "sklearn.svm.SVC".
        estimator_kwargs : dict[str, Any] | None, optional
            Параметры для конструктора оценщика (если estimator не задан).
        param_space : dict[str, Any] | None, optional
            Пространство параметров в формате DSL.
        cv : int | Iterable | None, optional (default=5)
            Разбиение для перекрёстной проверки.
        scoring : str | callable | dict | None, optional
            Метрика/метрики sklearn. Поддерживаются мульти-метрики; тогда refit может быть именем метрики.
        n_jobs : int | None, optional
            Число потоков для sklearn-поисковиков.
        refit : bool | str, optional (default=True)
            Поведение refit в sklearn-поисковиках. Для мульти-метрик — имя основной метрики.
        random_state : int | None, optional
            Перекрывает random_state фабрики.
        n_iter : int | None, optional
            Число итераций для RandomizedSearchCV (по умолчанию 50).
        n_trials : int | None, optional
            Число итераций для Optuna (по умолчанию 50).
        sampler : Any, optional
            Самплер Optuna.
        pruner : Any, optional
            Прунер Optuna.
        study_direction : str, optional (default="maximize")
            Направление оптимизации Optuna.
        timeout : int | None, optional
            Лимит времени для Optuna (интеграция/ручной режим).
        callbacks : Iterable[Callable] | None, optional
            Колбэки для Optуна.

        Возвращаемые значения
        ---------------------
        Any
            Экземпляр GridSearchCV/RandomizedSearchCV/OptunaSearchCV.
            Если OptunaSearchCV недоступен, возвращает кортеж:
            ("optuna-manual", config_dict).

        Исключения
        ----------
        UnknownSearchMethodError
            Неизвестный метод поиска.
        InvalidSearchSpaceError
            Некорректное пространство параметров для выбранного метода.
        OptionalDependencyError
            Отсутствуют необходимые опциональные зависимости (optuna/scipy).
        ValueError
            Ошибки валидации параметров конструктора оценщика.
        ImportError
            Не найден класс оценщика при импортe по пути.
        """
        from .errors import UnknownSearchMethodError

        key = normalize_name(method)
        cls, est = self._resolve_estimator(estimator, estimator_path, estimator_kwargs)

        eff_rs = random_state if random_state is not None else self.random_state

        if key in {"grid", "gridsearchcv"}:
            grid = space_to_grid(param_space or {})
            return GridSearchCV(
                estimator=est, param_grid=grid, cv=cv, scoring=scoring, n_jobs=n_jobs, refit=refit
            )

        if key in {"random", "randomizedsearchcv"}:
            dists = space_to_random_distributions(param_space or {})
            return RandomizedSearchCV(
                estimator=est,
                param_distributions=dists,
                n_iter=int(n_iter or 50),
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                refit=refit,
                random_state=eff_rs,
            )

        if key in {"optuna", "optunasearchcv"}:
            if not _HAS_OPTUNA:
                raise OptionalDependencyError(
                    "Вы выбрали 'optuna', но пакет 'optuna' не установлен. Установите: pip install optuna"
                )

            if _OptunaSearchCV is not None:
                # 1) конвертируем твой DSL → optuna.distributions
                dists = _space_to_optuna_dists(param_space or {})

                # 2) создаём OptunaSearchCV
                return _OptunaSearchCV(
                    estimator=est,
                    param_distributions=dists,
                    cv=cv,
                    scoring=scoring,
                    n_trials=int(n_trials or 50),
                    n_jobs=n_jobs,
                    refit=refit,
                    random_state=eff_rs,
                )

            # Ручной режим: возвращаем конфиг для run_optuna_manual(...)
            return (
                "optuna-manual",
                {
                    "estimator_class": cls,
                    "param_space": param_space or {},
                    "cv": cv,
                    "scoring": scoring,
                    "n_trials": int(n_trials or 50),
                    "sampler": sampler,
                    "pruner": pruner,
                    "study_direction": study_direction,
                    "fit_params": None,
                    "fixed_params": None,
                    "timeout": timeout,
                    "callbacks": callbacks,
                },
            )

        raise UnknownSearchMethodError(f"Неизвестный метод поиска: {method!r}")


def run_optuna_manual(
    estimator_class: Union[type, Callable[..., BaseEstimator]],
    grid: ParamSpace,
    X_train: Any,
    y_train: Any,
    *,
    scoring: ScoreArg | None = None,
    cv: Optional[Union[int, Iterable]] = 5,
    n_trials: int = 50,
    study_direction: str = "maximize",
    sampler: Any = None,
    pruner: Any = None,
    fit_params: Optional[Dict[str, Any]] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
    timeout: Optional[int] = None,
    callbacks: Optional[Iterable[Callable[..., Any]]] = None,
):
    """
    Запускает ручную оптимизацию Optuna с автоматически сгенерированной objective-функцией.

    Параметры
    ---------
    estimator_class : type | Callable[..., BaseEstimator]
        Класс оценщика или фабрика для создания экземпляров модели.
    grid : dict[str, Any]
        Пространство параметров в формате DSL.
    X_train : Any
        Обучающие признаки.
    y_train : Any
        Обучающие ответы/метки.
    scoring : str | callable | None, optional (default=None)
        Метрика для cross_val_score.
    cv : int | Iterable | None, optional (default=5)
        Разбиение для перекрёстной проверки.
    n_trials : int, optional (default=50)
        Число итераций оптимизации.
    study_direction : str, optional (default="maximize")
        Направление оптимизации: "maximize" или "minimize".
    sampler : Any, optional
        Самплер Optuna.
    pruner : Any, optional
        Прунер Optuna.
    fit_params : dict[str, Any] | None, optional
        Дополнительные параметры для .fit(...). Для sklearn>=1.4 используйте явный цикл CV.
    fixed_params : dict[str, Any] | None, optional
        Фиксированные параметры конструктора модели.
    timeout : int | None, optional
        Лимит времени оптимизации в секундах.
    callbacks : Iterable[Callable] | None, optional
        Колбэки для study.optimize(...).

    Возвращаемые значения
    ---------------------
    optuna.study.Study
        Объект исследования Optuna с лучшими параметрами и логом испытаний.

    Исключения
    ----------
    OptionalDependencyError
        Optuna не установлена.
    """
    if not _HAS_OPTUNA:
        raise OptionalDependencyError(
            "Для использования run_optuna_manual требуется 'optuna'. Установите: pip install optuna"
        )
    objective = gen_objective(
        estimator_class=estimator_class,
        grid=grid,
        X_train=X_train,
        y_train=y_train,
        scoring=scoring,
        cv=cv,
        fit_params=fit_params,
        greater_is_better=(study_direction == "maximize"),
        fixed_params=fixed_params,
    )
    study = _optuna.create_study(direction=study_direction, sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=list(callbacks or []))
    return study
