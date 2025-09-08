import importlib
from pathlib import Path
import pytest

from AutomationML import utils as U
from AutomationML.errors import OptionalDependencyError


# Топ-левел функция — должна корректно "стрингифицироваться"
def top_level_callable(**kw):
    return 123


def test_normalize_name_basic_and_case_insensitive():
    assert U.normalize_name("LinearRegression") == "linearregression"
    assert U.normalize_name("lOgIsTiCrEgReSsIoN") == "logisticregression"
    assert U.normalize_name("XGB_Classifier-01") == "xgb_classifier-01"


def test_normalize_name_none_and_empty():
    assert U.normalize_name(None) == ""
    assert U.normalize_name("") == ""


def test_import_from_path_success():
    obj = U.import_from_path("json.dumps")
    import json
    assert obj is json.dumps


def test_import_from_path_bad_module_raises_optional_dep():
    with pytest.raises(OptionalDependencyError):
        U.import_from_path("definitely_not_a_real_module.Foo")


def test_import_from_path_bad_attr_raises_import_error():
    with pytest.raises(ImportError):
        U.import_from_path("json.DoesNotExist")


def test_import_from_path_bad_value_raises_value_error():
    with pytest.raises(ValueError):
        U.import_from_path("NoDotHere")


def test_validate_kwargs_reports_invalid_and_suggestions():
    class A:
        def __init__(self, n_estimators=100, alpha=1.0):
            self.n_estimators = n_estimators
            self.alpha = alpha

    err = U.validate_kwargs(A, {"n_estimator": 10, "alfa": 0.5})
    assert isinstance(err, str)
    assert "Недопустимые параметры" in err
    assert "n_estimator" in err and "alfa" in err
    assert "Подсказки по возможным опечаткам" in err
    assert ("n_estimators" in err) or ("alpha" in err)


def test_validate_kwargs_all_valid_returns_none():
    class B:
        def __init__(self, x=1, y=2):
            self.x, self.y = x, y

    assert U.validate_kwargs(B, {"x": 10, "y": 20}) is None


def test_validate_kwargs_allows_var_kwargs_returns_none():
    class C:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    assert U.validate_kwargs(C, {"foo": 1, "bar": 2}) is None


def test_validate_kwargs_signature_failure_returns_none(monkeypatch):
    def boom(_):
        raise ValueError("no signature")

    # utils импортирует inspect на модульном уровне, патчим его
    monkeypatch.setattr(U.inspect, "signature", boom)

    class D:
        def __init__(self, a=1):
            self.a = a

    assert U.validate_kwargs(D, {"zzz": 2}) is None


def test_stringify_target_top_level_callable_success():
    s = U.stringify_target(top_level_callable)
    assert isinstance(s, str)
    assert top_level_callable.__name__ in s
    assert top_level_callable.__module__ in s


def test_stringify_target_nested_callable_returns_none():
    def outer():
        def inner():
            return 1
        return inner

    nested = outer()
    assert U.stringify_target(nested) is None


def test_stringify_target_lambda_returns_none():
    assert U.stringify_target(lambda: None) is None  # noqa: E731


def test_stringify_target_import_failure_returns_none(monkeypatch):
    modname = top_level_callable.__module__
    real_import = importlib.import_module

    def fake_import(name):
        if name == modname:
            raise ImportError("boom")
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    assert U.stringify_target(top_level_callable) is None


def test_json_safe_primitives_and_nonserializable():
    assert U.json_safe(10) == 10
    assert U.json_safe({"a": 1}) == {"a": 1}

    val = {"path": Path("somewhere"), "s": {1, 2, 3}}
    safe = {k: U.json_safe(v) for k, v in val.items()}
    assert isinstance(safe["path"], str)
    assert "Path(" in safe["path"]
    assert isinstance(safe["s"], str)


def test_maybe_inject_random_state_injects_when_supported_and_missing():
    class WithRS:
        def __init__(self, random_state=None, a=1):
            pass

    out = U.maybe_inject_random_state({}, WithRS, 42)
    assert out.get("random_state") == 42


def test_maybe_inject_random_state_does_not_override_existing():
    class WithRS:
        def __init__(self, random_state=None, a=1):
            pass

    out = U.maybe_inject_random_state({"random_state": 7}, WithRS, 42)
    assert out.get("random_state") == 7  # не перезаписываем


def test_maybe_inject_random_state_skips_when_not_supported():
    class NoRS:
        def __init__(self, a=1):
            pass

    out = U.maybe_inject_random_state({"a": 2}, NoRS, 99)
    assert "random_state" not in out
    assert out["a"] == 2


def test_maybe_inject_random_state_signature_failure(monkeypatch):
    def boom(_):
        raise TypeError("no sig")

    monkeypatch.setattr(U.inspect, "signature", boom)

    class AnyC:
        def __init__(self, a=1):
            pass

    original = {"a": 10}
    out = U.maybe_inject_random_state(original, AnyC, 123)
    assert out == original
