# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025
"""
Вспомогательные функции: безопасный импорт по строковому пути и валидация параметров.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List, Iterable
import importlib
import inspect
import difflib
import json

from .errors import OptionalDependencyError, InvalidSearchSpaceError


def import_from_path(path: str) -> Any:
    """
    Импортирует объект по строковому пути вида "package.module.ClassName".

    Параметры
    ---------
    path : str
        Полный путь к импортируемому объекту.

    Возвращаемые значения
    ---------------------
    Any
        Найденный объект (как правило, класс или функция).

    Исключения
    ----------
    OptionalDependencyError
        Модуль не найден (например, отсутствует пакет).
    ImportError
        Атрибут не найден в модуле.
    ValueError
        Путь не содержит имени модуля (нет точки).
    """
    module_path, _, attr_name = path.rpartition(".")
    if not module_path:
        raise ValueError(f"Неверный путь импорта: '{path}'")
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise OptionalDependencyError(
            f"Не удалось импортировать модуль '{module_path}' для '{path}'.\n"
            f"Вероятно, пакет не установлен. Пример установки:\n"
            f"pip install {module_path.split('.')[0]}"
        ) from e
    try:
        return getattr(module, attr_name)
    except AttributeError as e:
        raise ImportError(f"В модуле '{module_path}' не найден объект '{attr_name}'.") from e


def validate_kwargs(constructor: Any, params: Dict[str, Any]) -> Optional[str]:
    """
    Проверяет допустимость параметров по сигнатуре конструктора.

    Если у конструктора есть **kwargs, проверка пропускается.

    Параметры
    ---------
    constructor : Any
        Конструктор (класс или функция).
    params : dict[str, Any]
        Пары «имя параметра → значение», предполагаемые к передаче.

    Возвращаемые значения
    ---------------------
    str | None
        None — всё корректно; иначе строка с описанием ошибок и подсказками.
    """
    try:
        sig = inspect.signature(constructor)
    except (TypeError, ValueError):
        return None

    allowed = set()
    has_var_kw = False
    for name, p in sig.parameters.items():
        if p.kind in (p.POSITIONAL_ONLY, p.VAR_POSITIONAL):
            continue
        if p.kind == p.VAR_KEYWORD:
            has_var_kw = True
        else:
            allowed.add(name)

    if has_var_kw:
        return None

    invalid = [k for k in params.keys() if k not in allowed]
    if not invalid:
        return None

    suggestions: Dict[str, List[str]] = {}
    for k in invalid:
        close = difflib.get_close_matches(k, sorted(allowed), n=3, cutoff=0.6)
        if close:
            suggestions[k] = close

    parts = [f"Недопустимые параметры: {', '.join(sorted(invalid))}."]
    if suggestions:
        hint_lines = [f"  - {k}: возможно, имелось в виду: {', '.join(v)}" for k, v in suggestions.items()]
        parts.append("Подсказки по возможным опечаткам:\n" + "\n".join(hint_lines))
    parts.append(f"Допустимые параметры: {', '.join(sorted(allowed)) or '—'}.")
    return "\n".join(parts)


def maybe_inject_random_state(params: Dict[str, Any], constructor: Any, random_state: int) -> Dict[str, Any]:
    """
    Добавляет random_state в параметры, если конструктор его поддерживает и параметр не задан.

    Параметры
    ---------
    params : dict[str, Any]
        Исходные параметры.
    constructor : Any
        Конструктор (класс или функция).
    random_state : int
        Значение для подстановки.

    Возвращаемые значения
    ---------------------
    dict[str, Any]
        Копия словаря параметров с возможной подстановкой random_state.
    """
    out = dict(params or {})
    try:
        sig = inspect.signature(constructor)
        if "random_state" in sig.parameters and "random_state" not in out:
            out["random_state"] = random_state
    except (TypeError, ValueError):
        pass
    return out


def normalize_name(name: str) -> str:
    """
    Нормализует произвольное имя к виду, удобному для сравнений и ключей реестра.

    Параметры
    ---------
    name : str
        Произвольная строка.

    Возвращаемые значения
    ---------------------
    str
        Строка в нижнем регистре без изменений символов.
    """
    return (name or "").lower()


def stringify_target(target: Any) -> Optional[str]:
    """
    Возвращает импортируемый путь "module.qualname" для вызываемого объекта, если это возможно.

    Параметры
    ---------
    target : Any
        Вызываемый объект (функция/класс).

    Возвращаемые значения
    ---------------------
    str | None
        Строка "module.qualname" для топ-левел объектов; None для лямбд, вложенных функций и т. п.
    """
    if isinstance(target, str):
        return target
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if not module or not qualname or "<locals>" in qualname:
        return None
    try:
        mod = importlib.import_module(module)
        obj = mod
        for part in qualname.split("."):
            obj = getattr(obj, part)
        if obj is target:
            return f"{module}.{qualname}"
    except Exception:
        return None
    return None


def json_safe(value: Any) -> Any:
    """
    Приводит значение к виду, безопасному для json.dumps.

    Параметры
    ---------
    value : Any
        Произвольное значение.

    Возвращаемые значения
    ---------------------
    Any
        Исходное значение, если сериализуется; иначе строка repr(value).
    """
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _is_range_spec(spec: Any) -> bool:
    """
    True, если spec — DSL-диапазон: кортеж ( "int"|"float", low, high, [opts] ).
    """
    return (
        isinstance(spec, tuple)
        and len(spec) >= 3
        and isinstance(spec[0], str)
        and spec[0].lower() in {"int", "float"}
    )

def is_primitive_sequence(value: Any) -> bool:
    """
    Проверяет, является ли значение последовательностью примитивов (str|int|float|bool).

    ВАЖНО: DSL-диапазоны вида ("int"|"float", low, high, {opts}) НЕ считаются
    «примитивной последовательностью», чтобы их корректно обрабатывали space_to_grid/random.
    """
    if _is_range_spec(value):
        return False
    return isinstance(value, (list, tuple)) and all(isinstance(v, (str, int, float, bool)) for v in value)


def space_to_grid(space: Dict[str, Any]) -> Dict[str, Iterable[Any]]:
    """
    Преобразует DSL к формату param_grid для GridSearchCV.

    Правила
    -------
    — Категориальные параметры: возвращаются как списки значений.
    — Диапазоны ("int"/"float"): обязателен opts['step'] для дискретизации.
      Для int формируется range(...). Для float — равномерная сетка с шагом step.
    — Callable-значения не поддерживаются.

    Исключения
    ----------
    InvalidSearchSpaceError
        Диапазон без шага, некорректные границы/шаг или попытка передать callable.
    """
    out: Dict[str, Iterable[Any]] = {}
    for k, spec in (space or {}).items():
        if is_primitive_sequence(spec):
            out[k] = list(spec)
            continue

        if callable(spec):
            raise InvalidSearchSpaceError(
                f"Параметр '{k}' задан как callable и не может быть использован в GridSearchCV."
            )

        if isinstance(spec, tuple) and len(spec) >= 3 and isinstance(spec[0], str):
            kind = spec[0].lower()
            low, high = spec[1], spec[2]
            opts = spec[3] if len(spec) >= 4 and isinstance(spec[3], dict) else {}
            step = opts.get("step", None)
            if step is None:
                raise InvalidSearchSpaceError(
                    f"Параметр '{k}' задан как диапазон для GridSearchCV, но не указан 'step'. "
                    f"Задайте дискретизацию через opts['step'] или используйте метод 'random'/'optuna'."
                )

            try:
                if kind == "int":
                    low_i, high_i, step_i = int(low), int(high), int(step)
                    if low_i > high_i:
                        raise InvalidSearchSpaceError(f"Параметр '{k}': low должен быть <= high.")
                    if step_i <= 0:
                        raise InvalidSearchSpaceError(f"Параметр '{k}': step должен быть > 0.")
                    vals = list(range(low_i, high_i + 1, step_i))
                    if not vals or vals[-1] != high_i:
                        vals.append(high_i)
                    out[k] = vals

                elif kind == "float":
                    low_f, high_f, step_f = float(low), float(high), float(step)
                    if low_f > high_f:
                        raise InvalidSearchSpaceError(f"Параметр '{k}': low должен быть <= high.")
                    if step_f <= 0.0:
                        raise InvalidSearchSpaceError(f"Параметр '{k}': step должен быть > 0.")
                    n = max(1, int(round((high_f - low_f) / step_f)))
                    vals = [low_f + i * step_f for i in range(n + 1)]
                    if not vals or vals[-1] < high_f - 1e-15:
                        vals.append(high_f)
                    out[k] = vals

                else:
                    raise InvalidSearchSpaceError(
                        f"Параметр '{k}': неизвестный тип диапазона '{kind}'."
                    )
            except (TypeError, ValueError):
                raise InvalidSearchSpaceError(
                    f"Параметр '{k}': некорректный step/границы для диапазона."
                )
            continue

        out[k] = [spec]
    return out


def space_to_random_distributions(space: Dict[str, Any]) -> Dict[str, Any]:
    """
    Преобразует DSL к формату param_distributions для RandomizedSearchCV.

    Правила
    -------
    — Категориальные → списки.
    — Диапазоны без step:
         * int  -> scipy.stats.randint(low, high+1)
         * float + log=True -> scipy.stats.loguniform(low, high)
         * float + log=False -> scipy.stats.uniform(low, high-low)
       Для int c log=True и без step лог-шкала игнорируется (используется randint).
    — Диапазоны со step → дискретные списки значений.
    — Callable-значения не поддерживаются.

    Параметры
    ---------
    space : dict[str, Any]
        Пространство параметров в формате DSL.

    Возвращаемые значения
    ---------------------
    dict[str, Any]
        Словарь param_distributions, совместимый с RandomizedSearchCV.

    Исключения
    ----------
    OptionalDependencyError
        Отсутствует scipy.stats при необходимости распределений.
    InvalidSearchSpaceError
        Некорректные диапазоны или попытка передать callable.
    """
    try:
        import scipy.stats as st  # локальный импорт, чтобы не трогать глобально зависимости
    except Exception as e:  # pragma: no cover
        st = None

    out: Dict[str, Any] = {}
    for k, spec in (space or {}).items():
        if is_primitive_sequence(spec):
            out[k] = list(spec)
            continue
        if callable(spec):
            raise InvalidSearchSpaceError(
                f"Параметр '{k}' задан как callable и не может быть использован в RandomizedSearchCV."
            )
        if isinstance(spec, tuple) and len(spec) >= 3 and isinstance(spec[0], str):
            kind = spec[0].lower()
            low, high = spec[1], spec[2]
            opts = spec[3] if len(spec) >= 4 and isinstance(spec[3], dict) else {}
            step = opts.get("step", None)
            log = bool(opts.get("log", False))

            if step is not None:
                if kind == "int":
                    low_i, high_i, step_i = int(low), int(high), int(step)
                    if low_i > high_i:
                        raise InvalidSearchSpaceError(f"Параметр '{k}': low должен быть <= high.")
                    if step_i <= 0:
                        raise InvalidSearchSpaceError(f"Параметр '{k}': step должен быть > 0.")
                    vals = list(range(low_i, high_i + 1, step_i))
                    if not vals or vals[-1] != high_i:
                        vals.append(high_i)
                    out[k] = vals
                elif kind == "float":
                    low_f, high_f, step_f = float(low), float(high), float(step)
                    if low_f > high_f:
                        raise InvalidSearchSpaceError(f"Параметр '{k}': low должен быть <= high.")
                    if step_f <= 0.0:
                        raise InvalidSearchSpaceError(f"Параметр '{k}': step должен быть > 0.")
                    n = max(1, int(round((high_f - low_f) / step_f)))
                    vals = [low_f + i * step_f for i in range(n + 1)]
                    if not vals or vals[-1] < high_f - 1e-15:
                        vals.append(high_f)
                    out[k] = vals
                else:
                    raise InvalidSearchSpaceError(f"Параметр '{k}': неизвестный тип диапазона '{kind}'.")
                continue

            if st is None:
                raise OptionalDependencyError(
                    "Для RandomizedSearchCV с непрерывными распределениями требуется 'scipy'. "
                    "Установите пакет, например: pip install scipy"
                )

            if kind == "int":
                out[k] = st.randint(int(low), int(high) + 1)
            elif kind == "float":
                if log:
                    if float(low) <= 0:
                        raise InvalidSearchSpaceError(
                            f"Параметр '{k}': для loguniform нижняя граница должна быть > 0."
                        )
                    out[k] = st.loguniform(float(low), float(high))
                else:
                    out[k] = st.uniform(float(low), float(high) - float(low))
            else:
                raise InvalidSearchSpaceError(f"Параметр '{k}': неизвестный тип диапазона '{kind}'.")
            continue
        out[k] = [spec]
    return out
