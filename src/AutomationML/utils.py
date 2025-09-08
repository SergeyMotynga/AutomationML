# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Motynga Sergey/OMSTU
"""
Вспомогательные функции: безопасный импорт по строковому пути и валидация параметров.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List
import importlib
import inspect
import difflib
import json

from .errors import OptionalDependencyError


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
