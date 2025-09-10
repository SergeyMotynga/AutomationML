# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Motynga Sergey/OMSTU
"""
Конструктор конвейеров признаков и модели (`PipelineBuilder`).

Назначение
----------
Единый интерфейс для сборки `sklearn`/`imblearn`-совместимого `Pipeline` по
декларативной спецификации шагов без ручных импортов. Возвращаемый объект —
готовый `Pipeline`, который можно использовать через `.fit()`/`.predict()`.

Соглашения
----------
— Шаги задаются в виде экземпляра, класса/фабрики (callable) или строкового пути
  к классу вида "package.module.ClassName" (импорт выполняется лениво).
— Для ребалансировки классов поддерживаются короткие алиасы семплеров: "undersample",
  "smote", "combine". Эти шаги требуют пакета `imbalanced-learn`.
— Порядок по умолчанию: scaler → sampler → дополнительные шаги → model.
— Все параметры шага валидируются по сигнатуре конструктора с подсказками при опечатках.

Безопасность
------------
При построении по строковым путям выполняются импорты модулей. Не используйте
пользовательские спецификации из непроверённых источников.

Пример использования
--------------------
from AutomationML.pipeline import PipelineBuilder

# быстрый сценарий: масштабирование + undersample + модель
pb = PipelineBuilder(
    model="sklearn.linear_model.LogisticRegression",
    scaler=True,
    sample=True,
    sampler="undersample",
    sampling_strategy=0.3,
)
pipe = pb.build()
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# продвинутый сценарий: явные объекты и параметры
from sklearn.preprocessing import StandardScaler
pb = PipelineBuilder(
    model="sklearn.ensemble.RandomForestClassifier",
    scaler=StandardScaler(with_mean=False),
    sampler={"alias": "smote", "params": {"k_neighbors": 5, "sampling_strategy": 0.5}},
)
pipe = pb.build()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Callable
import inspect

from .errors import OptionalDependencyError
from .utils import import_from_path, validate_kwargs
from sklearn.pipeline import Pipeline as SkPipeline

Spec = Union[str, Callable[..., Any], Any, Dict[str, Any]]


class PipelineBuilder:
    """
    Конструктор `Pipeline` c поддержкой строковых импортов и алиасов семплеров.

    Параметры
    ---------
    model : Spec
        Обязательный финальный шаг (классификатор или регрессор). Принимает
        экземпляр, класс/фабрику (callable) или строковый путь к классу.
    scaler : bool | Spec, по умолчанию None
        Масштабирование признаков. ``True`` разворачивается в
        ``sklearn.preprocessing.StandardScaler``. Можно передать экземпляр, callable
        или строковый путь.
    sample : bool, по умолчанию True
        Включает шаг балансировки классов (только для задач классификации).
    sampler : str | Spec, по умолчанию "undersample"
        Семплер дисбаланса. Допускаются алиасы: ``"undersample"``, ``"smote"``,
        ``"combine"`` либо полная спека как для остальных шагов.
    sampling_strategy : Any, по умолчанию 0.3
        Параметр семплера; добавляется если подходит по сигнатуре конструктора.
    extra_steps : Sequence[dict], по умолчанию ()
        Дополнительные шаги до модели. Элемент: {"name": str, "obj": Spec, "params": dict?}.
        Порядок соблюдается как в списке.
    random_state : int | None, по умолчанию 42
        Базовый random_state, автоматически подставляется, если поддерживается конструктором
        и не задан явно.
    n_jobs : int | None, по умолчанию 1
        Аналогично подставляется у шагов препроцессинга/семплинга; для шага 'model' не применяется.

    Примечания
    ----------
    Возвращается `imblearn.pipeline.Pipeline`, если в шагах есть `sampler`, иначе —
    `sklearn.pipeline.Pipeline`. Это необходимо для корректной работы с `fit_resample`.
    """

    _SAMPLER_ALIASES = {
        "undersample": "imblearn.under_sampling.RandomUnderSampler",
        "smote": "imblearn.over_sampling.SMOTE",
        "combine": "imblearn.combine.SMOTEENN",
    }

    def __init__(
        self,
        *,
        model: Spec,
        scaler: Optional[Spec] = None,
        sample: bool = True,
        sampler: Union[str, Spec] = "undersample",
        sampling_strategy: Optional[Any] = 0.3,
        extra_steps: Sequence[Dict[str, Any]] = (),
        random_state: Optional[int] = 42,
        n_jobs: Optional[int] = 1,
    ) -> None:
        """
        Инициализирует билдер с выбранной схемой шагов.

        Допускаются разные формы спецификаций шагов: экземпляр, callable/класс,
        строковый путь к классу или словарь {"obj": <spec>, "params": {...}}.

        Параметры см. в описании класса.
        """
        self._model_spec = model
        self._scaler_spec = scaler
        self._sample = bool(sample)
        self._sampler_spec = sampler
        self._sampling_strategy = sampling_strategy
        self._extra_steps = list(extra_steps or [])
        self._random_state = random_state
        self._n_jobs = n_jobs

    def build(self):
        """
        Собирает и возвращает `Pipeline` на основании заданной конфигурации.

        Возвращаемые значения
        ---------------------
        sklearn.pipeline.Pipeline | imblearn.pipeline.Pipeline
            Готовый конвейер признаков и модели.

        Исключения
        ----------
        ValueError
            Не указан обязательный шаг `model` или неверная спецификация шага.
        OptionalDependencyError
            Требуется пакет `imbalanced-learn` для шага `sampler`.
        TypeError
            Параметры шага не проходят валидацию по сигнатуре конструктора.

        Замечания
        ---------
        Если включён шаг `sampler`, используется `imblearn.pipeline.Pipeline`,
        иначе — `sklearn.pipeline.Pipeline`.
        """
        steps: List[Tuple[str, Any]] = []

        if self._scaler_spec is True:
            scaler_spec: Spec = "sklearn.preprocessing.StandardScaler"
        else:
            scaler_spec = self._scaler_spec
        if scaler_spec is not None:
            steps.append(("scaler", self._create_instance("scaler", scaler_spec, {})))

        if self._sample and self._sampler_spec is not None:
            sampler_params: Dict[str, Any] = {}
            if self._sampling_strategy is not None:
                sampler_params["sampling_strategy"] = self._sampling_strategy
            steps.append(("sampler", self._create_sampler(sampler_params)))

        for item in self._extra_steps:
            name = item.get("name")
            obj = item.get("obj")
            params = dict(item.get("params", {}) or {})
            if not name or not isinstance(name, str):
                raise ValueError("Каждый шаг в extra_steps должен иметь строковое поле 'name'.")
            steps.append((name, self._create_instance(name, obj, params)))

        steps.append(("model", self._create_instance("model", self._model_spec, {})))

        if any(n == "sampler" for n, _ in steps):
            try:
                from imblearn.pipeline import Pipeline as ImbPipeline
            except ModuleNotFoundError as e:
                raise OptionalDependencyError(
                    "Для использования шага семплинга требуется пакет imbalanced-learn.\n"
                    "Установка: pip install imbalanced-learn"
                ) from e
            return ImbPipeline(steps)
        return SkPipeline(steps)

    def _create_sampler(self, base_params: Dict[str, Any]) -> Any:
        """
        Создаёт экземпляр шага `sampler` с учётом алиасов и пользовательских параметров.

        Параметры
        ---------
        base_params : dict
            Параметры, которые всегда добавляются к семплеру (например, `sampling_strategy`).

        Возвращает
        ----------
        Any
            Инициализированный объект семплера (`RandomUnderSampler`, `SMOTE`, `SMOTEENN` и т.п.).

        Исключения
        ----------
        ValueError
            Неизвестный алиас семплера в словарной спецификации.
        """
        spec = self._sampler_spec
        if isinstance(spec, str) and spec in self._SAMPLER_ALIASES:
            spec = self._SAMPLER_ALIASES[spec]
        elif isinstance(spec, dict) and "alias" in spec:
            alias = spec.get("alias")
            if alias not in self._SAMPLER_ALIASES:
                raise ValueError(f"Неизвестный алиас семплера: {alias!r}.")
            spec = {"obj": self._SAMPLER_ALIASES[alias], "params": {**(spec.get("params", {}) or {}), **base_params}}
        return self._create_instance("sampler", spec, base_params)

    def _create_instance(self, name: str, spec: Spec, params: Dict[str, Any]) -> Any:
        """
        Нормализует спецификацию шага и создаёт его экземпляр.

        Поддерживаемые формы `spec`:
        — экземпляр шага;
        — callable/класс (будет вызван с параметрами);
        — строка с путём к классу (импорт и инициализация);
        — словарь вида {"obj": <spec>, "params": {...}}.

        Параметры
        ---------
        name : str
            Имя шага в итоговом конвейере (например, "scaler", "sampler", "model").
        spec : Spec
            Спецификация шага.
        params : dict
            Дополнительные параметры, объединяемые с параметрами из словарной спеки.

        Возвращает
        ----------
        Any
            Инициализированный объект шага.

        Исключения
        ----------
        ValueError
            Пустая спецификация или отсутствует ключ `obj` в словарной спеке.
        TypeError
            Параметры не соответствуют сигнатуре конструктора; сообщение содержит подсказки.
        """
        if spec is None:
            raise ValueError(f"Шаг '{name}' не может быть создан: пустая спецификация.")

        if isinstance(spec, dict):
            obj = spec.get("obj")
            if obj is None:
                raise ValueError(f"Для шага '{name}' требуется ключ 'obj' в спецификации.")
            merge = dict(spec.get("params", {}) or {})
            params = {**merge, **params}
            spec = obj

        if isinstance(spec, str):
            constructor = import_from_path(spec)
        elif inspect.isclass(spec) or callable(spec):
            constructor = spec
        else:
            # передан готовый экземпляр
            instance = spec
            self._maybe_set_common_defaults(instance, params, is_model=(name == "model"))
            self._maybe_set_params(instance, params)
            return instance

        # автоподстановка параметров; для 'model' n_jobs не подставляется
        params = self._inject_common_defaults(name, constructor, params)

        err = validate_kwargs(constructor, params)
        if err:
            raise TypeError(
                f"Параметры шага '{name}' некорректны относительно конструктора {constructor}.\n{err}"
            )
        try:
            return constructor(**params)
        except TypeError as e:
            bad = ", ".join(sorted(params.keys()))
            raise TypeError(
                f"Не удалось создать шаг '{name}' из {constructor}. Переданные параметры: {bad}.\nИсходная ошибка: {e}"
            ) from e

    def _inject_common_defaults(self, name: str, constructor: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Подставляет общие значения (`random_state`, `n_jobs`) в параметры конструктора шага.

        Правила
        -------
        — `random_state` подставляется, если есть в сигнатуре конструктора и отсутствует в `params`.
        — `n_jobs` подставляется аналогично, **но никогда не для шага 'model'**.
        — Если сигнатура недоступна (динамический конструктор), возвращает параметры как есть.

        Параметры
        ---------
        name : str
            Имя шага (нужно, чтобы исключить подстановку `n_jobs` для 'model').
        constructor : Any
            Конструктор шага (класс или фабрика).
        params : dict
            Пользовательские параметры, которые имеют приоритет над автоподстановкой.

        Возвращает
        ----------
        dict
            Итоговый набор параметров, готовый для передачи в конструктор.
        """
        out = dict(params or {})
        try:
            sig = inspect.signature(constructor)
        except (TypeError, ValueError):
            return out
        if self._random_state is not None and "random_state" in sig.parameters and "random_state" not in out:
            out["random_state"] = self._random_state
        if name != "model" and self._n_jobs is not None and "n_jobs" in sig.parameters and "n_jobs" not in out:
            out["n_jobs"] = self._n_jobs
        return out

    @staticmethod
    def _maybe_set_params(instance: Any, params: Dict[str, Any]) -> None:
        """
        Аккуратно применяет параметры к уже созданному экземпляру шага.

        Если у экземпляра есть метод `set_params`, параметры будут установлены.
        Не вызывает исключений, если `params` пуст или метод отсутствует.

        Параметры
        ---------
        instance : Any
            Готовый экземпляр шага.
        params : dict
            Параметры для установки.
        """
        if not params:
            return
        if hasattr(instance, "set_params"):
            instance.set_params(**params)

    def _maybe_set_common_defaults(self, instance: Any, params: Dict[str, Any], is_model: bool = False) -> None:
        """
        Устанавливает общие значения (`random_state`, `n_jobs`) для уже созданного экземпляра.

        Правила
        -------
        — Проверяются доступные параметры через `get_params(deep=False)`.
        — Значения выставляются только если пользователь их не задал в `params`.
        — Для `model` параметр `n_jobs` **никогда** не устанавливается.
        — Для любого экземпляра значения не переопределяются, если уже заданы у объекта.

        Параметры
        ---------
        instance : Any
            Готовый экземпляр шага.
        params : dict
            Пользовательские параметры (имеют приоритет).
        is_model : bool, по умолчанию False
            Признак того, что экземпляр является финальной моделью.
        """
        if not hasattr(instance, "get_params"):
            return
        try:
            allowed = instance.get_params(deep=False)
        except Exception:
            return

        to_set: Dict[str, Any] = {}

        # random_state выставляем только если:
        # 1) параметр поддерживается; 2) пользователь не задал его в params;
        # 3) у инстанса он не установлен (None).
        if (
            self._random_state is not None
            and "random_state" in allowed
            and "random_state" not in params
            and getattr(instance, "random_state", None) is None
        ):
            to_set["random_state"] = self._random_state

        # n_jobs не выставляем для модели; и в целом — только если у инстанса он None
        if (
            not is_model
            and self._n_jobs is not None
            and "n_jobs" in allowed
            and "n_jobs" not in params
            and getattr(instance, "n_jobs", None) is None
        ):
            to_set["n_jobs"] = self._n_jobs

        if to_set:
            self._maybe_set_params(instance, to_set)
