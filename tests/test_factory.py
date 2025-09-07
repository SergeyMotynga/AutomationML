import importlib
from pathlib import Path
import json
import os
import pytest

from AutomationML import factory as factory_module
from AutomationML.factory import (
    ModelFactory,
    ModelSpec,
    ModelNotFoundError,
    TaskMismatchError,
    OptionalDependencyError,
)

def test_invalid_param_validation_message_contains_suggestions():
    f = ModelFactory()
    with pytest.raises(TypeError) as ei:
        f.create("RandomForestClassifier", n_estimator=10)
    msg = str(ei.value)
    assert "Недопустимые параметры" in msg
    assert "n_estimator" in msg
    assert "Подсказки по возможным опечаткам" in msg
    assert "n_estimators" in msg

def test_polynomial_regression_defaults_set_inner_n_jobs():
    f = ModelFactory()
    pipe = f.create("PolynomialRegression")
    steps = dict(pipe.named_steps)
    inner_lr = steps["LinearRegression"]
    assert getattr(inner_lr, "n_jobs", None) == 1

def test_list_models_contains_expected():
    f = ModelFactory()
    names = f.list_models()
    assert "LinearRegression" in names
    assert "RandomForestClassifier" in names
    assert "PolynomialRegression" in names

def test_list_models_task_filter():
    f = ModelFactory()
    clfs = set(f.list_models(task="clf"))
    regs = set(f.list_models(task="reg"))
    assert "LogisticRegression" in clfs and "LinearRegression" not in clfs
    assert "LinearRegression" in regs and "LogisticRegression" not in regs

def test_get_help_for_string_target_includes_core_fields():
    f = ModelFactory()
    info = f.get_help("RandomForestRegressor")
    assert "Модель: RandomForestRegressor" in info
    assert "Задача:" in info
    assert "Дефолты:" in info
    assert "Источник:" in info

def test_get_help_for_callable_target_shows_factory_name():
    f = ModelFactory()

    def dummy_factory(**kw):
        class C:
            ...
        return C()

    f.register_model("DummyCallable", target=dummy_factory, defaults={}, task="reg", doc="d")
    info = f.get_help("DummyCallable")
    assert "<фабрика dummy_factory>" in info

def test_create_with_overrides_applied():
    f = ModelFactory()
    lin = f.create("LinearRegression", fit_intercept=False, n_jobs=2)
    assert lin.fit_intercept is False
    assert getattr(lin, "n_jobs", None) == 2

def test_create_random_forest_overrides_n_jobs_and_estimators():
    f = ModelFactory()
    rf = f.create("RandomForestClassifier", n_estimators=123, n_jobs=-1, random_state=777)
    assert rf.n_estimators == 123
    assert rf.n_jobs == -1
    assert rf.random_state == 777

def test_polynomial_regression_pipeline_and_inner_params():
    f = ModelFactory()
    pipe = f.create(
        "PolynomialRegression",
        degree=3,
        include_bias=False,
        linear_kwargs={"fit_intercept": True, "n_jobs": 3},
    )
    from sklearn.pipeline import Pipeline
    assert isinstance(pipe, Pipeline)
    steps = dict(pipe.named_steps)
    assert "PolynomialFeatures" in steps and "LinearRegression" in steps
    assert steps["PolynomialFeatures"].degree == 3
    assert steps["PolynomialFeatures"].include_bias is False
    assert getattr(steps["LinearRegression"], "n_jobs", None) == 3

def test_stacking_models_create_ok():
    f = ModelFactory()
    sc = f.create("StackingClassifier", passthrough=True)
    sr = f.create("StackingRegressor")
    from sklearn.ensemble import StackingClassifier, StackingRegressor
    assert isinstance(sc, StackingClassifier)
    assert isinstance(sr, StackingRegressor)

def test_register_overwrite_and_unregister_model():
    f = ModelFactory()
    f.register_model("TmpModel", target="sklearn.tree.DecisionTreeClassifier", defaults={}, task="clf", doc="v1")
    assert "TmpModel" in f.list_models()
    f.register_model(
        "TmpModel",
        target="sklearn.tree.DecisionTreeClassifier",
        defaults={"max_depth": 2},
        task="clf",
        doc="v2",
        overwrite=True,
    )
    tmp = f.create("TmpModel")
    assert getattr(tmp, "max_depth", None) == 2
    f.unregister_model("TmpModel")
    assert "TmpModel" not in f.list_models()
    with pytest.raises(ModelNotFoundError):
        f.unregister_model("TmpModel")

def test_expected_task_mismatch():
    f = ModelFactory()
    with pytest.raises(TaskMismatchError):
        f.create("LogisticRegression", expected_task="reg")

def test_invalid_param_validation_message_contains_hints():
    f = ModelFactory()
    with pytest.raises(TypeError) as ei:
        f.create("DecisionTreeClassifier", max_iter=100)
    msg = str(ei.value)
    assert "Недопустимые параметры" in msg
    assert "max_iter" in msg
    assert "Допустимые параметры" in msg

def test_unknown_model_name_suggestions_present():
    f = ModelFactory()
    with pytest.raises(ModelNotFoundError) as ei:
        f.create("RandomForrestClassifier")
    msg = str(ei.value)
    assert "Похожие:" in msg
    assert "RandomForestClassifier" in msg

def test_unknown_model_name_without_suggestions():
    f = ModelFactory()
    with pytest.raises(ModelNotFoundError) as ei:
        f.create("XYZ_foo_bar_baz_qux")
    msg = str(ei.value)
    assert "Похожие:" not in msg

def test_optional_dependency_error_on_missing_package(monkeypatch):
    real_import = importlib.import_module
    def fake_import_module(name):
        if name == "xgboost":
            raise ModuleNotFoundError("No module named 'xgboost'")
        return real_import(name)
    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    f = ModelFactory()
    with pytest.raises(OptionalDependencyError):
        f.create("XGBClassifier")

def test_import_error_when_class_missing():
    f = ModelFactory()
    f.register_model("BogusModel", target="sklearn.linear_model.DoesNotExist", defaults={}, task="reg")
    with pytest.raises(ImportError):
        f.create("BogusModel")

def test_value_error_on_bad_import_path():
    f = ModelFactory()
    f.register_model("BadPathModel", target="LinearRegression", defaults={}, task="reg")
    with pytest.raises(ValueError):
        f.create("BadPathModel")

def test_validate_kwargs_skips_when_var_kw_allows_unknowns():
    class KW:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    f = ModelFactory()
    f.register_model("KwModel", target=KW, defaults={"a": 1}, task="reg")
    m = f.create("KwModel", unknown_param=123)
    assert isinstance(m, KW)
    assert m.kwargs["unknown_param"] == 123
    assert m.kwargs["a"] == 1

def test_typeerror_after_validation_is_wrapped():
    class Boom:
        def __init__(self, **kwargs):
            raise TypeError("boom")
    f = ModelFactory()
    f.register_model("BoomModel", target=Boom, defaults={"p": 1}, task="reg")
    with pytest.raises(TypeError) as ei:
        f.create("BoomModel", q=2)
    assert "Исходная ошибка" in str(ei.value)

def test_inspect_signature_failure_branch(monkeypatch):
    class Simple:
        def __init__(self, a=1):
            self.a = a
    def fake_signature(obj):
        raise ValueError("no signature")
    monkeypatch.setattr(factory_module.inspect, "signature", fake_signature)
    f = ModelFactory()
    f.register_model("SimpleModel", target=Simple, defaults={"a": 1}, task="reg")
    s = f.create("SimpleModel", a=2)
    assert s.a == 2

def test_case_insensitive_lookup_create_and_help():
    f = ModelFactory()
    lin = f.create("linearregression", fit_intercept=False)
    assert lin.fit_intercept is False
    info = f.get_help("linearregression")
    assert "Модель: LinearRegression" in info  # каноническое имя в справке

def test_has_model_true_false_and_unregister_builtin_protection():
    f = ModelFactory()
    assert f.has_model("LINEARREGRESSION")
    with pytest.raises(ValueError):
        f.unregister_model("LinearRegression")  # встроенная запись защищена
    # удаление встроенной записи с force=True
    f.unregister_model("LinearRegression", force=True)
    assert not f.has_model("LinearRegression")

def test_dump_registry_json_contains_format_version_and_json_safe_defaults():
    f = ModelFactory()
    # регистрируем модель со сложным значением в defaults (не JSON-сериализуемым)
    from sklearn.tree import DecisionTreeClassifier
    f.register_model(
        "MyTree",
        target=DecisionTreeClassifier,
        defaults={"max_depth": 3, "path": Path("somewhere")},  # Path не сериализуется стандартно
        task="clf",
        eager_validate=False,  # чтобы не падать на неизвестном параметре "path"
    )
    data = f.dump_registry_json(include_builtins=False)
    assert '"format_version": 1' in data
    assert "MyTree" in data
    assert "Path(" in data  # repr(Path(...)) будет содержать 'PosixPath' или 'WindowsPath'

def test_dump_strict_raises_on_lambda_target():
    f = ModelFactory()
    f.register_model("LambdaModel", target=lambda **k: None, defaults={}, task="reg")
    with pytest.raises(ValueError) as ei:
        _ = f.dump(strict=True)
    assert "LambdaModel" in str(ei.value)

def test_save_and_load_registry_roundtrip(tmp_path):
    f = ModelFactory()
    f.register_model(
        "ElasticNet",
        target="sklearn.linear_model.ElasticNet",
        defaults={"alpha": 0.1, "l1_ratio": 0.7, "max_iter": 10000},
        task="reg",
        eager_validate=True,
    )
    path = tmp_path / "automationml_registry.json"
    saved = f.save_registry(str(path), include_builtins=False)
    assert saved == str(path)
    assert path.exists() and path.stat().st_size > 0

    f2 = ModelFactory()
    f2.load_registry_file(str(path), eager_validate=True)
    assert "ElasticNet" in f2.list_models(task="reg")
    m = f2.create("ElasticNet", alpha=0.2)
    from sklearn.linear_model import ElasticNet
    assert isinstance(m, ElasticNet)
    assert m.alpha == 0.2

def test_load_registry_file_rejects_bad_version(tmp_path):
    payload = {"format_version": 999, "items": []}
    p = tmp_path / "bad_version.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    f = ModelFactory()
    with pytest.raises(ValueError) as ei:
        f.load_registry_file(str(p))
    assert "Неподдерживаемая версия формата" in str(ei.value)

def test_load_registry_file_accepts_list_payload(tmp_path):
    items = [{
        "model_name": "DT",
        "target": "sklearn.tree.DecisionTreeClassifier",
        "defaults": {"max_depth": 2},
        "task": "clf",
        "doc": "short",
    }]
    p = tmp_path / "list_payload.json"
    p.write_text(json.dumps(items), encoding="utf-8")
    f = ModelFactory()
    f.load_registry_file(str(p), eager_validate=True)
    dt = f.create("DT")
    from sklearn.tree import DecisionTreeClassifier
    assert isinstance(dt, DecisionTreeClassifier)
    assert dt.max_depth == 2

def test_strict_validation_catches_unknown_param_post_init():
    class Loose:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def get_params(self, deep=False):
            return {"a": 1}  # допустим только параметр 'a'
    f = ModelFactory(strict_validation=True)
    f.register_model("Loose", target=Loose, defaults={"a": 1}, task="reg")
    with pytest.raises(TypeError) as ei:
        f.create("Loose", b=2)  # b не распознан get_params
    assert "Параметры не распознаны" in str(ei.value)

def test_strict_validation_disabled_allows_unknown_param_post_init():
    class Loose:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def get_params(self, deep=False):
            return {"a": 1}
    f = ModelFactory(strict_validation=False)
    f.register_model("Loose", target=Loose, defaults={"a": 1}, task="reg")
    m = f.create("Loose", b=2)
    assert isinstance(m, Loose)
    assert m.kwargs["b"] == 2

def test_register_model_eager_validate_catches_bad_defaults():
    f = ModelFactory()
    with pytest.raises(TypeError) as ei:
        f.register_model(
            "BadDefaults",
            target="sklearn.tree.DecisionTreeClassifier",
            defaults={"max_iters": 10},  # опечатка
            task="clf",
            eager_validate=True,
        )
    assert "Дефолтные параметры записи 'BadDefaults' некорректны" in str(ei.value)

def test_has_model_and_case_insensitive_after_unregister():
    f = ModelFactory()
    assert f.has_model("linearREGRESSION")
    f.unregister_model("LinearRegression", force=True)
    assert not f.has_model("linearregression")
