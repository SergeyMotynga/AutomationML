# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Motynga Sergey/OMSTU
"""
Фабрика моделей с ленивыми импортами и полными именами классов.

Назначение
----------
Единый интерфейс для создания экземпляров моделей машинного обучения
по строковому полному имени класса без ручных импортов. Импорты выполняются
лениво при первом создании соответствующей записи реестра.

Соглашения


----------
— Используются полные имена классов (например, "LinearRegression", "RandomForestClassifier").
— Гиперпараметры передаются напрямую через create(...); они перекрывают значения из defaults в спецификации.
— Опциональные зависимости (catboost, xgboost, lightgbm) импортируются только по запросу (при create()).
— Сообщения об ошибках и документация на русском.

Безопасность
------------
Функции загрузки реестра импортируют указанные объекты по строковым путям.
Не загружайте реестры из непроверенных источников.

Пример использования
--------------------
from AutomationML.factory import ModelFactory

factory = ModelFactory(random_state=42)
linear = factory.create("LinearRegression", fit_intercept=True)
forest = factory.create("RandomForestClassifier", n_estimators=300, n_jobs=-1)
svc = factory.create("SVC", kernel="rbf", C=2.0)

# Регистрация пользовательской модели
factory.register_model(
    "ElasticNet",
    target="sklearn.linear_model.ElasticNet",
    defaults={"alpha": 0.1, "l1_ratio": 0.7, "max_iter": 10_000},
    task="reg",
    eager_validate=True,
)

# Сохранение и загрузка реестра
factory.save_registry("automationml_registry.json")
factory2 = ModelFactory()
factory2.load_registry_file("automationml_registry.json", eager_validate=True)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, List, Union, Tuple
import importlib
import inspect
import difflib
import json
import threading

from .errors import ModelNotFoundError, OptionalDependencyError, TaskMismatchError
from .utils import (
    validate_kwargs as _validate_kwargs,
    normalize_name,
    stringify_target,
    json_safe,
)

EstimatorConstructor = Callable[..., Any]
ImportPath = str


@dataclass
class ModelSpec:
    """
    Спецификация записи в реестре фабрики.

    Параметры
    ---------
    target : str | Callable[..., Any]
        Полный путь к классу/фабрике (например, "sklearn.linear_model.LogisticRegression")
        или непосредственно вызываемый объект.
    defaults : dict[str, Any]
        Дефолтные параметры конструктора. Переопределяются при вызове create(...).
    task : {"clf", "reg", None}
        Тип задачи: "clf" — классификация, "reg" — регрессия, None — не указан.
    doc : str
        Краткое описание модели.

    Атрибуты
    --------
    target : str | Callable[..., Any]
    defaults : dict[str, Any]
    task : str | None
    doc : str
    """
    target: Union[ImportPath, EstimatorConstructor]
    defaults: Dict[str, Any] = field(default_factory=dict)
    task: Optional[str] = None
    doc: str = ""

    def resolve(self) -> EstimatorConstructor:
        """
        Возвращает конструктор модели; при строковом target выполняет ленивый импорт.

        Возвращаемые значения
        ---------------------
        Callable[..., Any]
            Конструктор модели (класс или функция-фабрика).

        Исключения
        ----------
        OptionalDependencyError
            Модуль, содержащий целевой класс/фабрику, не установлен.
        ImportError
            Класс/фабрика не найдены в импортированном модуле.
        ValueError
            Некорректный путь импорта (без имени модуля).
        """
        if callable(self.target):
            return self.target

        module_path, _, attr_name = self.target.rpartition(".")
        if not module_path:
            raise ValueError(f"Неверный путь импорта: '{self.target}'")

        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as e:
            raise OptionalDependencyError(
                f"Не удалось импортировать модуль '{module_path}' для '{self.target}'.\n"
                f"Вероятно, пакет не установлен. Установка примера:\n"
                f"pip install {module_path.split('.')[0]}"
            ) from e

        try:
            obj = getattr(module, attr_name)
        except AttributeError as e:
            raise ImportError(
                f"В модуле '{module_path}' не найден объект '{attr_name}'."
            ) from e

        return obj


class ModelFactory:
    """
    Фабрика для создания экземпляров моделей по полным именам.

    Параметры
    ---------
    random_state : int, по умолчанию 42
        Базовое значение для стохастических моделей; используется в defaults там, где применимо.
    strict_validation : bool, по умолчанию True
        Дополнительная валидация через get_params после создания модели.
        Полезно для классов, принимающих **kwargs в конструкторе: выявляет опечатки в параметрах.
    """

    def __init__(self, random_state: int = 42, *, strict_validation: bool = True):
        """
        Инициализация фабрики и регистрация стандартного набора моделей.

        Параметры
        ---------
        random_state : int, по умолчанию 42
            Значение, которое будет подставлено в параметры моделей, поддерживающих random_state.
        strict_validation : bool, по умолчанию True
            Включение дополнительной проверки параметров через get_params.
        """
        self.random_state = random_state
        self.strict_validation = strict_validation
        self._lock = threading.RLock()
        # Ключ реестра — нормализованное имя, значение — кортеж (каноническое имя, спецификация)
        self._registry: Dict[str, Tuple[str, ModelSpec]] = {}
        # Набор встроенных имён (в нормализованном виде)
        self._builtin_names: set[str] = set()
        self._register_defaults()

    def has_model(self, model_name: str) -> bool:
        """
        Проверяет наличие записи в реестре по имени (без учёта регистра).

        Параметры
        ---------
        model_name : str
            Имя модели, которое требуется проверить.

        Возвращаемые значения
        ---------------------
        bool
            True, если запись существует; иначе False.
        """
        key = normalize_name(model_name)
        with self._lock:
            return key in self._registry

    def register_model(
        self,
        model_name: str,
        target: Union[str, EstimatorConstructor],
        *,
        defaults: Optional[Dict[str, Any]] = None,
        task: Optional[str] = None,
        doc: str = "",
        overwrite: bool = False,
        eager_validate: bool = False,
        _internal: bool = False,
    ) -> None:
        """
        Регистрирует модель в реестре (для внутренних и пользовательских записей).

        Параметры
        ---------
        model_name : str
            Каноническое имя модели (как будет отображаться в списках и справке).
        target : str | Callable[..., Any]
            Полный путь к классу/функции-фабрике или сам вызываемый объект.
        defaults : dict[str, Any] | None, по умолчанию None
            Дефолтные параметры конструктора; перекрываются при вызове create(...).
        task : {"clf", "reg", None}, по умолчанию None
            Тип задачи для фильтрации и базовой проверки expected_task.
        doc : str, по умолчанию ""
            Краткое описание модели.
        overwrite : bool, по умолчанию False
            Разрешить перезапись существующей записи.
        eager_validate : bool, по умолчанию False
            Немедленно проверить корректность defaults относительно конструктора.
        _internal : bool, по умолчанию False
            Маркер встроенной записи (защита от удаления/перезаписи без явного разрешения).

        Возвращаемые значения
        ---------------------
        None

        Исключения
        ----------
        ValueError
            Некорректное значение параметра task.
        ModelNotFoundError
            Имя уже занято (при overwrite=False) или попытка перезаписать встроенную запись без разрешения.
        OptionalDependencyError | ImportError | TypeError
            Возможны при eager_validate=True (ошибки импорта или несоответствие параметров).
        """
        if task not in (None, "clf", "reg"):
            raise ValueError("Недопустимое значение 'task'. Ожидается: None, 'clf' или 'reg'.")

        key = normalize_name(model_name)
        with self._lock:
            if (key in self._builtin_names) and (not overwrite) and (not _internal):
                raise ModelNotFoundError(
                    f"Имя '{model_name}' принадлежит встроенной записи. "
                    f"Разрешить перезапись можно, указав overwrite=True."
                )
            if (key in self._registry) and (not overwrite):
                raise ModelNotFoundError(
                    f"Модель '{model_name}' уже существует в реестре. "
                    f"Для перезаписи установите overwrite=True."
                )

            spec = ModelSpec(target=target, defaults=defaults or {}, task=task, doc=doc)

            if eager_validate:
                constructor = spec.resolve()
                err = _validate_kwargs(constructor, spec.defaults)
                if err:
                    raise TypeError(f"Дефолтные параметры записи '{model_name}' некорректны:\n{err}")

            self._registry[key] = (model_name, spec)
            if _internal:
                self._builtin_names.add(key)

    def unregister_model(self, model_name: str, *, force: bool = False) -> None:
        """
        Удаляет запись из реестра по имени.

        Параметры
        ---------
        model_name : str
            Имя модели, которую требуется удалить.
        force : bool, по умолчанию False
            Разрешить удаление встроенной записи.

        Возвращаемые значения
        ---------------------
        None

        Исключения
        ----------
        ModelNotFoundError
            Запись отсутствует в реестре.
        ValueError
            Попытка удалить встроенную запись без force=True.
        """
        key = normalize_name(model_name)
        with self._lock:
            if key not in self._registry:
                raise ModelNotFoundError(f"Невозможно удалить: модель '{model_name}' не найдена.")
            if (key in self._builtin_names) and (not force):
                raise ValueError(f"'{model_name}' — встроенная запись. Для удаления укажите force=True.")
            del self._registry[key]
            self._builtin_names.discard(key)

    def list_models(self, task: Optional[str] = None) -> List[str]:
        """
        Возвращает отсортированный список зарегистрированных моделей.

        Параметры
        ---------
        task : {"clf", "reg", None}, по умолчанию None
            Фильтрация по типу задачи: "clf" — классификация, "reg" — регрессия, None — без фильтра.

        Возвращаемые значения
        ---------------------
        list[str]
            Список канонических имён моделей (включая встроенные и пользовательские записи).
        """
        with self._lock:
            out: List[str] = []
            for canon_name, spec in self._registry.values():
                if task is None or spec.task == task:
                    out.append(canon_name)
            return sorted(out)

    def _get_spec(self, model_name: str) -> Tuple[str, ModelSpec]:
        """
        Возвращает каноническое имя и спецификацию по произвольному регистру имени.

        Параметры
        ---------
        model_name : str
            Имя модели (любой регистр).

        Возвращаемые значения
        ---------------------
        tuple[str, ModelSpec]
            Кортеж (каноническое имя, спецификация).

        Исключения
        ----------
        ModelNotFoundError
            Модель не найдена; сообщение включает подсказки ближайших совпадений.
        """
        key = normalize_name(model_name)
        with self._lock:
            item = self._registry.get(key)
            if item is not None:
                return item
            candidates = [canon for canon, _ in self._registry.values()]
            suggestions = difflib.get_close_matches(model_name, candidates, n=5, cutoff=0.6)
            hint = f" Похожие: {', '.join(suggestions)}." if suggestions else ""
            raise ModelNotFoundError(f"Модель '{model_name}' не найдена.{hint}")

    def get_help(self, model_name: str) -> str:
        """
        Возвращает краткую справку по модели: тип задачи, дефолты, источник и описание.

        Параметры
        ---------
        model_name : str
            Имя модели, для которой требуется вывести информацию.

        Возвращаемые значения
        ---------------------
        str
            Текстовый блок со сведениями о модели.

        Исключения
        ----------
        ModelNotFoundError
            Модель не найдена в реестре.
        """
        canon, spec = self._get_spec(model_name)
        target_desc = spec.target if isinstance(spec.target, str) else f"<фабрика {spec.target.__name__}>"
        return (
            f"Модель: {canon}\n"
            f"Задача: {spec.task or 'не указана'}\n"
            f"Дефолты: {spec.defaults or '{}'}\n"
            f"Источник: {target_desc}\n"
            f"Описание: {spec.doc or '—'}"
        )

    def create(self, model_name: str, expected_task: Optional[str] = None, **overrides) -> Any:
        """
        Создаёт экземпляр модели по имени из реестра с валидацией параметров.

        Параметры
        ---------
        model_name : str
            Имя модели (ключ в реестре; регистр не важен).
        expected_task : {"clf", "reg", None}, по умолчанию None
            Ожидаемый тип задачи. Если отличается от типа в спецификации, будет выброшен TaskMismatchError.
        **overrides
            Дополнительные параметры конструктора; перекрывают значения из defaults в спецификации.

        Возвращаемые значения
        ---------------------
        Any
            Инициализированный экземпляр модели.

        Исключения
        ----------
        ModelNotFoundError
            Модель отсутствует в реестре.
        TaskMismatchError
            Тип задачи не соответствует ожидаемому.
        OptionalDependencyError
            Требуемый пакет не установлен (для catboost/xgboost/lightgbm и т. п.).
        ImportError
            Целевой класс/фабрика не найден(а) в модуле.
        TypeError
            Обнаружены недопустимые параметры конструктора (на этапе проверки или инициализации).
        """
        canon, spec = self._get_spec(model_name)

        if expected_task is not None and spec.task is not None and expected_task != spec.task:
            raise TaskMismatchError(
                f"Модель '{canon}' относится к задаче '{spec.task}', а ожидалась '{expected_task}'."
            )

        constructor = spec.resolve()
        params = {**spec.defaults, **overrides}

        validation_error = _validate_kwargs(constructor, params)
        if validation_error:
            raise TypeError(
                f"Не удалось создать модель '{canon}' из-за неверных параметров.\n{validation_error}"
            )

        try:
            model = constructor(**params)
        except TypeError as e:
            bad = ", ".join(sorted(params.keys()))
            raise TypeError(
                f"Не удалось создать модель '{canon}' с указанными параметрами.\n"
                f"Переданные параметры: {bad}\n"
                f"Исходная ошибка: {e}"
            ) from e

        # Дополнительная валидация через get_params (если включена)
        if self.strict_validation and inspect.isclass(constructor) and hasattr(model, "get_params"):
            try:
                allowed = set(model.get_params(deep=False).keys())
            except Exception:
                allowed = None

            if allowed is not None:
                unknown = [k for k in params.keys() if k not in allowed]
                if unknown:
                    suggestions: Dict[str, List[str]] = {}
                    for k in unknown:
                        close = difflib.get_close_matches(k, sorted(allowed), n=3, cutoff=0.6)
                        if close:
                            suggestions[k] = close

                    parts = [f"Параметры не распознаны моделью после инициализации: {', '.join(sorted(unknown))}."]
                    if suggestions:
                        hint_lines = [
                            f"  - {k}: возможно, имелось в виду: {', '.join(v)}"
                            for k, v in suggestions.items()
                        ]
                        parts.append("Подсказки по возможным опечаткам:\n" + "\n".join(hint_lines))
                    parts.append(f"Допустимые параметры: {', '.join(sorted(allowed)) or '—'}.")

                    raise TypeError("\n".join(parts))

        return model

    def dump_registry(self, *, include_builtins: bool = False, strict: bool = False) -> List[Dict[str, Any]]:
        """
        Возвращает элементы реестра в сериализуемом виде (для сохранения в JSON).

        Параметры
        ---------
        include_builtins : bool, по умолчанию False
            Включать ли встроенные записи в результат.
        strict : bool, по умолчанию False
            Требовать сериализацию всех записей. Если True и есть несериализуемые цели (callable
            без импортируемого пути), будет выброшен ValueError.

        Возвращаемые значения
        ---------------------
        list[dict]
            Список элементов вида:
            {
              "model_name": str,
              "target": str,           # импортируемый путь
              "defaults": dict,        # значения приведены к JSON-безопасному виду
              "task": "clf" | "reg" | None,
              "doc": str
            }

        Исключения
        ----------
        ValueError
            В режиме strict=True присутствуют записи, которые нельзя сериализовать.
        """
        with self._lock:
            items: List[Dict[str, Any]] = []
            skipped: List[str] = []
            for key, (canon, spec) in self._registry.items():
                if (key in self._builtin_names) and (not include_builtins):
                    continue
                s_target = stringify_target(spec.target)
                if s_target is None:
                    skipped.append(canon)
                    continue
                safe_defaults = {k: json_safe(v) for k, v in spec.defaults.items()}
                items.append({
                    "model_name": canon,
                    "target": s_target,
                    "defaults": safe_defaults,
                    "task": spec.task,
                    "doc": spec.doc,
                })
            if strict and skipped:
                raise ValueError(
                    "Невозможно сериализовать следующие записи (target не имеет импортируемого пути): "
                    + ", ".join(sorted(skipped))
                )
            return items

    def dump_registry_json(
        self,
        *,
        include_builtins: bool = False,
        strict: bool = False,
        ensure_ascii: bool = False,
        indent: int = 2,
    ) -> str:
        """
        Возвращает JSON-строку со всем реестром.

        Параметры
        ---------
        include_builtins : bool, по умолчанию False
            Включать ли встроенные записи.
        strict : bool, по умолчанию False
            Требовать сериализацию всех записей.
        ensure_ascii : bool, по умолчанию False
            Управляет экранированием не-ASCII символов.
        indent : int, по умолчанию 2
            Визуальное форматирование JSON.

        Возвращаемые значения
        ---------------------
        str
            JSON-строка формата {"format_version": 1, "items": [...]}.

        Исключения
        ----------
        ValueError
            В режиме strict=True присутствуют записи, которые нельзя сериализовать.
        """
        payload = {"format_version": 1, "items": self.dump_registry(include_builtins=include_builtins, strict=strict)}
        return json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent)

    def dump(self, *, include_builtins: bool = False, strict: bool = False) -> str:
        """
        Короткий синоним dump_registry_json().
        """
        return self.dump_registry_json(include_builtins=include_builtins, strict=strict)

    def save_registry(
        self,
        path: str,
        *,
        include_builtins: bool = False,
        strict: bool = False,
        ensure_ascii: bool = False,
        indent: int = 2,
        encoding: str = "utf-8",
    ) -> str:
        """
        Сохраняет весь реестр в JSON-файл.

        Параметры
        ---------
        path : str
            Путь к целевому файлу (например, "automationml_registry.json").
        include_builtins : bool, по умолчанию False
            Включать ли встроенные записи.
        strict : bool, по умолчанию False
            Требовать сериализацию всех записей.
        ensure_ascii : bool, по умолчанию False
            Управляет экранированием не-ASCII символов.
        indent : int, по умолчанию 2
            Визуальное форматирование JSON.
        encoding : str, по умолчанию "utf-8"
            Кодировка выходного файла.

        Возвращаемые значения
        ---------------------
        str
            Путь к записанному файлу.

        Исключения
        ----------
        ValueError
            В режиме strict=True присутствуют записи, которые нельзя сериализовать.
        """
        data = self.dump_registry_json(include_builtins=include_builtins, strict=strict,
                                       ensure_ascii=ensure_ascii, indent=indent)
        with open(path, "w", encoding=encoding) as f:
            f.write(data)
        return path

    def load_registry(
        self,
        items: List[Dict[str, Any]],
        *,
        overwrite: bool = False,
        eager_validate: bool = False,
    ) -> None:
        """
        Загружает записи в реестр из списка словарей.

        Параметры
        ---------
        items : list[dict]
            Список элементов с ключами: model_name, target, defaults?, task?, doc?.
        overwrite : bool, по умолчанию False
            Разрешить перезапись существующих записей.
        eager_validate : bool, по умолчанию False
            Немедленно проверять корректность defaults относительно конструктора.

        Возвращаемые значения
        ---------------------
        None

        Исключения
        ----------
        OptionalDependencyError | ImportError | TypeError
            Возможны при eager_validate=True.
        """
        for it in items:
            self.register_model(
                it["model_name"],
                target=it["target"],
                defaults=it.get("defaults") or {},
                task=it.get("task"),
                doc=it.get("doc", ""),
                overwrite=overwrite,
                eager_validate=eager_validate,
            )

    def load_registry_file(
        self,
        path: str,
        *,
        overwrite: bool = False,
        eager_validate: bool = False,
        encoding: str = "utf-8",
    ) -> None:
        """
        Загружает реестр из JSON-файла, созданного save_registry() или совместимого формата.

        Параметры
        ---------
        path : str
            Путь к JSON-файлу.
        overwrite : bool, по умолчанию False
            Разрешить перезапись существующих записей.
        eager_validate : bool, по умолчанию False
            Немедленно проверять корректность defaults относительно конструктора.
        encoding : str, по умолчанию "utf-8"
            Кодировка файла.

        Возвращаемые значения
        ---------------------
        None

        Исключения
        ----------
        ValueError
            Неподдерживаемая версия формата реестра или неверная структура файла.
        OptionalDependencyError | ImportError | TypeError
            Возможны при eager_validate=True.
        """
        with open(path, "r", encoding=encoding) as f:
            payload = json.load(f)

        # Строгая проверка версии формата
        if isinstance(payload, dict):
            version = payload.get("format_version")
            if version is not None and version != 1:
                raise ValueError(f"Неподдерживаемая версия формата реестра: {version!r}. Ожидается 1.")
            items = payload.get("items", [])
        elif isinstance(payload, list):
            items = payload
        else:
            raise ValueError("Неверный формат файла реестра: ожидается объект или список.")

        self.load_registry(items, overwrite=overwrite, eager_validate=eager_validate)

    def _register_defaults(self) -> None:
        """
        Регистрирует стандартный набор моделей.

        Примечания
        ----------
        — Параметр n_jobs добавлен в defaults только у моделей, где он поддерживается; по умолчанию n_jobs=1.
        — Значение random_state подставляется там, где это уместно.
        """
        # Классификация
        self.register_model(
            "LogisticRegression",
            target="sklearn.linear_model.LogisticRegression",
            defaults={"random_state": self.random_state, "max_iter": 10_000},
            task="clf",
            doc="Логистическая регрессия (классификация).",
            _internal=True,
        )
        self.register_model(
            "SVC",
            target="sklearn.svm.SVC",
            defaults={"random_state": self.random_state},
            task="clf",
            doc="SVM-классификатор (ядровой).",
            _internal=True,
        )
        self.register_model(
            "KNeighborsClassifier",
            target="sklearn.neighbors.KNeighborsClassifier",
            defaults={},
            task="clf",
            doc="KNN-классификатор.",
            _internal=True,
        )
        self.register_model(
            "GaussianNB",
            target="sklearn.naive_bayes.GaussianNB",
            defaults={},
            task="clf",
            doc="Наивный Байес (гауссовский).",
            _internal=True,
        )
        self.register_model(
            "DecisionTreeClassifier",
            target="sklearn.tree.DecisionTreeClassifier",
            defaults={"random_state": self.random_state},
            task="clf",
            doc="Дерево решений (классификация).",
            _internal=True,
        )
        self.register_model(
            "RandomForestClassifier",
            target="sklearn.ensemble.RandomForestClassifier",
            defaults={"random_state": self.random_state, "n_jobs": 1},
            task="clf",
            doc="Случайный лес (классификация).",
            _internal=True,
        )
        self.register_model(
            "GradientBoostingClassifier",
            target="sklearn.ensemble.GradientBoostingClassifier",
            defaults={"random_state": self.random_state},
            task="clf",
            doc="Градиентный бустинг (классификация) из sklearn.",
            _internal=True,
        )
        self.register_model(
            "BaggingClassifier",
            target="sklearn.ensemble.BaggingClassifier",
            defaults={"random_state": self.random_state, "n_jobs": 1},
            task="clf",
            doc="Бэггинг (классификация).",
            _internal=True,
        )
        self.register_model(
            "CatBoostClassifier",
            target="catboost.CatBoostClassifier",
            defaults={"verbose": False, "random_state": self.random_state},
            task="clf",
            doc="CatBoost (классификация).",
            _internal=True,
        )
        self.register_model(
            "XGBClassifier",
            target="xgboost.XGBClassifier",
            defaults={"random_state": self.random_state, "n_estimators": 200, "n_jobs": 1},
            task="clf",
            doc="XGBoost (классификация).",
            _internal=True,
        )
        self.register_model(
            "LGBMClassifier",
            target="lightgbm.LGBMClassifier",
            defaults={"random_state": self.random_state, "n_estimators": 200, "n_jobs": 1},
            task="clf",
            doc="LightGBM (классификация).",
            _internal=True,
        )

        # Регрессия
        self.register_model(
            "LinearRegression",
            target="sklearn.linear_model.LinearRegression",
            defaults={"fit_intercept": True, "copy_X": True, "n_jobs": 1, "positive": False},
            task="reg",
            doc="Классическая линейная регрессия (МНК).",
            _internal=True,
        )
        self.register_model(
            "Ridge",
            target="sklearn.linear_model.Ridge",
            defaults={"alpha": 1.0, "solver": "auto", "positive": False},
            task="reg",
            doc="Гребневая регрессия (L2-регуляризация).",
            _internal=True,
        )
        self.register_model(
            "Lasso",
            target="sklearn.linear_model.Lasso",
            defaults={"alpha": 1.0, "max_iter": 10_000, "fit_intercept": True},
            task="reg",
            doc="Лассо-регрессия (L1-регуляризация).",
            _internal=True,
        )

        def polynomial_regression_factory(**params):
            """
            Создаёт Pipeline(PolynomialFeatures -> LinearRegression) для полиномиальной регрессии.

            Параметры
            ---------
            **params
                Параметры фабрики:
                - degree : int, по умолчанию 2 — степень полинома;
                - interaction_only : bool, по умолчанию False — только взаимодействия без степеней;
                - include_bias : bool, по умолчанию True — добавлять столбец единиц;
                - linear_kwargs : dict[str, Any], по умолчанию {} — параметры внутреннего LinearRegression
                  (по умолчанию в нём проставляется n_jobs=1, если не указан).

            Возвращаемые значения
            ---------------------
            sklearn.pipeline.Pipeline
                Сконфигурированный конвейер признаков и линейной регрессии.
            """
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression

            degree = params.pop("degree", 2)
            interaction_only = params.pop("interaction_only", False)
            include_bias = params.pop("include_bias", True)
            linear_kwargs = params.pop("linear_kwargs", {})
            linear_kwargs.setdefault("n_jobs", 1)

            return Pipeline(
                steps=[
                    ("PolynomialFeatures", PolynomialFeatures(
                        degree=degree,
                        interaction_only=interaction_only,
                        include_bias=include_bias,
                        order="C"
                    )),
                    ("LinearRegression", LinearRegression(**linear_kwargs)),
                ],
                memory=None,
                verbose=False,
            )

        self.register_model(
            "PolynomialRegression",
            target=polynomial_regression_factory,
            defaults={
                "degree": 2,
                "interaction_only": False,
                "include_bias": True,
                "linear_kwargs": {},
            },
            task="reg",
            doc="Полиномиальная регрессия как Pipeline(PolynomialFeatures -> LinearRegression).",
            _internal=True,
        )

        self.register_model(
            "SVR",
            target="sklearn.svm.SVR",
            defaults={},
            task="reg",
            doc="SVM-регрессор (ядровой).",
            _internal=True,
        )
        self.register_model(
            "KNeighborsRegressor",
            target="sklearn.neighbors.KNeighborsRegressor",
            defaults={},
            task="reg",
            doc="KNN-регрессор.",
            _internal=True,
        )
        self.register_model(
            "DecisionTreeRegressor",
            target="sklearn.tree.DecisionTreeRegressor",
            defaults={"random_state": self.random_state},
            task="reg",
            doc="Дерево решений (регрессия).",
            _internal=True,
        )
        self.register_model(
            "RandomForestRegressor",
            target="sklearn.ensemble.RandomForestRegressor",
            defaults={"random_state": self.random_state, "n_jobs": 1},
            task="reg",
            doc="Случайный лес (регрессия).",
            _internal=True,
        )
        self.register_model(
            "GradientBoostingRegressor",
            target="sklearn.ensemble.GradientBoostingRegressor",
            defaults={"random_state": self.random_state},
            task="reg",
            doc="Градиентный бустинг (регрессия) из sklearn.",
            _internal=True,
        )
        self.register_model(
            "BaggingRegressor",
            target="sklearn.ensemble.BaggingRegressor",
            defaults={"random_state": self.random_state, "n_jobs": 1},
            task="reg",
            doc="Бэггинг (регрессия).",
            _internal=True,
        )
        self.register_model(
            "CatBoostRegressor",
            target="catboost.CatBoostRegressor",
            defaults={"verbose": False, "random_state": self.random_state},
            task="reg",
            doc="CatBoost (регрессия).",
            _internal=True,
        )
        self.register_model(
            "XGBRegressor",
            target="xgboost.XGBRegressor",
            defaults={"random_state": self.random_state, "n_estimators": 300, "n_jobs": 1},
            task="reg",
            doc="XGBoost (регрессия).",
            _internal=True,
        )
        self.register_model(
            "LGBMRegressor",
            target="lightgbm.LGBMRegressor",
            defaults={"random_state": self.random_state, "n_estimators": 300, "n_jobs": 1},
            task="reg",
            doc="LightGBM (регрессия).",
            _internal=True,
        )

        def stacking_classifier_factory(**params):
            """
            Создаёт StackingClassifier с разумными базовыми моделями по умолчанию.

            Параметры
            ---------
            **params
                Параметры конструктора StackingClassifier.
                По умолчанию:
                - estimators : list[tuple[str, estimator]]
                    DecisionTreeClassifier и KNeighborsClassifier из текущей фабрики;
                - final_estimator : sklearn.linear_model.LogisticRegression
                    Логистическая регрессия с max_iter=10_000 и текущим random_state.

            Возвращаемые значения
            ---------------------
            sklearn.ensemble.StackingClassifier
                Сконфигурированная стековая модель классификации.
            """
            from sklearn.ensemble import StackingClassifier
            from sklearn.linear_model import LogisticRegression
            base_estimators = [
                ("DecisionTreeClassifier", self.create("DecisionTreeClassifier")),
                ("KNeighborsClassifier", self.create("KNeighborsClassifier")),
            ]
            estimators = params.pop("estimators", base_estimators)
            final_estimator = params.pop(
                "final_estimator",
                LogisticRegression(max_iter=10_000, random_state=self.random_state),
            )
            return StackingClassifier(estimators=estimators, final_estimator=final_estimator, **params)

        self.register_model(
            "StackingClassifier",
            target=stacking_classifier_factory,
            defaults={"n_jobs": 1},
            task="clf",
            doc="StackingClassifier с дефолтными базовыми моделями (DecisionTree + KNeighbors).",
            _internal=True,
        )

        def stacking_regressor_factory(**params):
            """
            Создаёт StackingRegressor с разумными базовыми моделями по умолчанию.

            Параметры
            ---------
            **params
                Параметры конструктора StackingRegressor.
                По умолчанию:
                - estimators : list[tuple[str, estimator]]
                    DecisionTreeRegressor и KNeighborsRegressor из текущей фабрики;
                - final_estimator : sklearn.svm.SVR
                    Регрессор опорных векторов.

            Возвращаемые значения
            ---------------------
            sklearn.ensemble.StackingRegressor
                Сконфигурированная стековая модель регрессии.
            """
            from sklearn.ensemble import StackingRegressor
            from sklearn.svm import SVR
            base_estimators = [
                ("DecisionTreeRegressor", self.create("DecisionTreeRegressor")),
                ("KNeighborsRegressor", self.create("KNeighborsRegressor")),
            ]
            estimators = params.pop("estimators", base_estimators)
            final_estimator = params.pop("final_estimator", SVR())
            return StackingRegressor(estimators=estimators, final_estimator=final_estimator, **params)

        self.register_model(
            "StackingRegressor",
            target=stacking_regressor_factory,
            defaults={"n_jobs": 1},
            task="reg",
            doc="StackingRegressor с дефолтными базовыми моделями (DecisionTree + KNeighbors).",
            _internal=True,
        )
