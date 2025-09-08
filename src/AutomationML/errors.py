# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Motynga Sergey/OMSTU
"""
Общие типы ошибок для AutomationML.
"""

class AutomationMLError(Exception):
    """Базовый класс ошибок библиотеки."""

class ModelNotFoundError(KeyError, AutomationMLError):
    """Модель с указанным именем отсутствует в реестре."""

class OptionalDependencyError(ImportError, AutomationMLError):
    """Требуемый внешний пакет не установлен."""

class TaskMismatchError(ValueError, AutomationMLError):
    """Несоответствие типа задачи (классификация/регрессия)."""
