import sys
import types
import builtins
import pytest

from AutomationML.pipeline import PipelineBuilder
from AutomationML.errors import OptionalDependencyError


class DummyTransformer:
    def __init__(self, n_jobs=None, random_state=None):
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def get_params(self, deep=False):
        return {"n_jobs": self.n_jobs, "random_state": self.random_state}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


class NoParamsTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class WithGetNoSet:
    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def get_params(self, deep=False):
        return {"n_jobs": self.n_jobs}


class DummyLooseCtor:
    def __init__(self, **kwargs):
        raise TypeError("ctor failure")


def test_without_sampler_returns_sklearn_pipeline_and_scaler_true():
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import StandardScaler

    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        scaler=True,
        sample=False,
    )
    pipe = pb.build()

    assert isinstance(pipe, SkPipeline)
    assert list(pipe.named_steps.keys()) == ["scaler", "model"]
    assert isinstance(pipe.named_steps["scaler"], StandardScaler)


def test_with_sampler_alias_smote_returns_imblearn_pipeline_and_sets_random_state():
    pytest.importorskip("imblearn")
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE

    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        scaler=True,
        sample=True,
        sampler="smote",
        sampling_strategy=0.3,
        random_state=123,
    )
    pipe = pb.build()
    assert isinstance(pipe, ImbPipeline)

    sampler = pipe.named_steps["sampler"]
    assert isinstance(sampler, SMOTE)
    assert getattr(sampler, "sampling_strategy") == 0.3
    assert getattr(sampler, "random_state") == 123


def test_sampler_dict_alias_params_merge_and_strategy():
    pytest.importorskip("imblearn")
    from imblearn.over_sampling import SMOTE

    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        scaler=True,
        sample=True,
        sampler={"alias": "smote", "params": {"k_neighbors": 5}},
        sampling_strategy=0.4,
        random_state=13,
    )
    pipe = pb.build()
    sm = pipe.named_steps["sampler"]
    assert isinstance(sm, SMOTE)
    assert getattr(sm, "k_neighbors") == 5
    assert getattr(sm, "sampling_strategy") == 0.4
    assert getattr(sm, "random_state") == 13


def test_sampler_full_path_string_spec():
    pytest.importorskip("imblearn")
    from imblearn.over_sampling import SMOTE

    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        scaler=True,
        sample=True,
        sampler="imblearn.over_sampling.SMOTE",
        sampling_strategy=0.2,
    )
    pipe = pb.build()
    sm = pipe.named_steps["sampler"]
    assert isinstance(sm, SMOTE)
    assert getattr(sm, "sampling_strategy") == 0.2


def test_sampler_class_spec():
    pytest.importorskip("imblearn")
    from imblearn.over_sampling import SMOTE

    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        scaler=True,
        sample=True,
        sampler=SMOTE,
        sampling_strategy=0.25,
        random_state=9,
    )
    pipe = pb.build()
    sm = pipe.named_steps["sampler"]
    assert isinstance(sm, SMOTE)
    assert getattr(sm, "sampling_strategy") == 0.25
    assert getattr(sm, "random_state") == 9


def test_sampler_without_strategy_is_ok():
    pytest.importorskip("imblearn")
    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        scaler=True,
        sample=True,
        sampler="undersample",
        sampling_strategy=None,
    )
    pipe = pb.build()
    assert "sampler" in pipe.named_steps


def test_extra_steps_order_and_defaults_applied():
    from sklearn.pipeline import Pipeline as SkPipeline

    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        scaler=True,
        sample=False,
        extra_steps=[{"name": "dummy", "obj": DummyTransformer}],
        random_state=7,
        n_jobs=99,
    )
    pipe = pb.build()
    assert isinstance(pipe, SkPipeline)
    assert list(pipe.named_steps.keys()) == ["scaler", "dummy", "model"]

    dummy = pipe.named_steps["dummy"]
    assert isinstance(dummy, DummyTransformer)
    assert dummy.random_state == 7
    assert dummy.n_jobs == 99


def test_extra_step_instance_preserves_existing_random_state_and_sets_n_jobs_if_none():
    inst = DummyTransformer(random_state=5, n_jobs=None)
    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        scaler=None,
        sample=False,
        extra_steps=[{"name": "dummy", "obj": inst}],
        random_state=7,
        n_jobs=3,
    )
    pipe = pb.build()
    dummy = pipe.named_steps["dummy"]
    assert dummy.random_state == 5
    assert dummy.n_jobs == 3


def test_extra_step_without_get_params_is_ignored_safely():
    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        scaler=None,
        sample=False,
        extra_steps=[{"name": "nop", "obj": NoParamsTransformer}],
        random_state=1,
        n_jobs=2,
    )
    pipe = pb.build()
    assert "nop" in pipe.named_steps


def test_extra_step_with_get_no_set_does_not_crash():
    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        scaler=None,
        sample=False,
        extra_steps=[{"name": "wg", "obj": WithGetNoSet}],
        n_jobs=4,
    )
    pipe = pb.build()
    assert "wg" in pipe.named_steps


def test_callable_constructor_injection_and_model_does_not_get_n_jobs():
    def make_dummy(n_jobs=None, random_state=None):
        return DummyTransformer(n_jobs=n_jobs, random_state=random_state)

    from sklearn.ensemble import RandomForestClassifier

    pb = PipelineBuilder(
        model=lambda random_state=None, n_jobs=None: RandomForestClassifier(
            random_state=random_state, n_jobs=n_jobs
        ),
        scaler=None,
        sample=False,
        extra_steps=[{"name": "dummy", "obj": make_dummy}],
        random_state=10,
        n_jobs=8,
    )
    pipe = pb.build()
    dummy = pipe.named_steps["dummy"]
    model = pipe.named_steps["model"]
    assert (dummy.random_state, dummy.n_jobs) == (10, 8)
    assert getattr(model, "random_state", None) == 10
    assert getattr(model, "n_jobs", None) != 8


def test_model_class_does_not_receive_n_jobs_but_may_receive_random_state():
    from sklearn.ensemble import RandomForestClassifier

    pb = PipelineBuilder(
        model=RandomForestClassifier,
        scaler=True,
        sample=False,
        random_state=42,
        n_jobs=12345,
    )
    pipe = pb.build()
    model = pipe.named_steps["model"]

    assert getattr(model, "random_state", None) == 42
    assert getattr(model, "n_jobs", None) != 12345


def test_model_instance_keeps_its_own_n_jobs_and_random_state():
    from sklearn.ensemble import RandomForestClassifier

    mdl = RandomForestClassifier(n_jobs=7, random_state=111)
    pb = PipelineBuilder(
        model=mdl,
        scaler=True,
        sample=False,
        n_jobs=999,
        random_state=222,
    )
    pipe = pb.build()
    model = pipe.named_steps["model"]
    assert model.n_jobs == 7
    assert model.random_state == 111


def test_invalid_sampler_alias_raises_value_error():
    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        scaler=True,
        sample=True,
        sampler={"alias": "unknown"},
    )
    with pytest.raises(ValueError):
        pb.build()


def test_extra_steps_without_name_raises_value_error():
    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        sample=False,
        extra_steps=[{"obj": DummyTransformer}],
    )
    with pytest.raises(ValueError):
        pb.build()


def test_missing_obj_in_dict_spec_raises_value_error():
    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        sample=False,
        extra_steps=[{"name": "bad", "params": {"k": 10}}],
    )
    with pytest.raises(ValueError):
        pb.build()


def test_params_validation_raises_type_error():
    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        sample=False,
        extra_steps=[{
            "name": "bad_scaler",
            "obj": "sklearn.preprocessing.StandardScaler",
            "params": {"foo": 1},
        }],
    )
    with pytest.raises(TypeError):
        pb.build()


def test_ctor_type_error_bubbles_with_clear_message():
    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        sample=False,
        extra_steps=[{"name": "loose", "obj": DummyLooseCtor}],
    )
    with pytest.raises(TypeError) as ei:
        pb.build()
    assert "ctor failure" in str(ei.value)


def test_value_error_when_spec_is_none_in_extra_step():
    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        sample=False,
        extra_steps=[{"name": "none", "obj": None}],
    )
    with pytest.raises(ValueError):
        pb.build()


def test_optional_dependency_error_when_imblearn_missing(monkeypatch):
    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("imblearn"):
            raise ModuleNotFoundError("No module named 'imblearn'")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        scaler=True,
        sample=True,
        sampler="undersample",
    )
    from AutomationML.errors import OptionalDependencyError
    with pytest.raises(OptionalDependencyError):
        pb.build()


def test_fit_predict_without_sampler():
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [0, 0, 1, 1]
    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        scaler=True,
        sample=False,
    )
    pipe = pb.build()
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert len(pred) == len(y)


def test_fit_predict_with_sampler_undersample():
    pytest.importorskip("imblearn")
    X = [[0.0], [0.1], [0.2], [5.0], [6.0], [7.0]]
    y = [0, 0, 0, 1, 1, 1]
    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        scaler=True,
        sample=True,
        sampler="undersample",
        sampling_strategy=0.5,
    )
    pipe = pb.build()
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert len(pred) == len(y)


# ---- ДОПОЛНИТЕЛЬНО ДЛЯ ПОКРЫТИЯ ----

def test_scaler_string_path_ok():
    from sklearn.pipeline import Pipeline as SkPipeline
    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        scaler="sklearn.preprocessing.MinMaxScaler",
        sample=False,
    )
    pipe = pb.build()
    assert isinstance(pipe, SkPipeline)
    assert list(pipe.named_steps.keys()) == ["scaler", "model"]
    assert pipe.named_steps["scaler"].__class__.__name__ == "MinMaxScaler"


def test_minimal_pipeline_only_model():
    from sklearn.pipeline import Pipeline as SkPipeline
    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        scaler=None,
        sample=False,
    )
    pipe = pb.build()
    assert isinstance(pipe, SkPipeline)
    assert list(pipe.named_steps.keys()) == ["model"]


def test_signature_failure_path_no_injection(monkeypatch):
    import inspect as _inspect

    def fake_sig(obj):
        raise ValueError("signature failed")

    monkeypatch.setattr(_inspect, "signature", fake_sig, raising=True)

    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        scaler="sklearn.preprocessing.MinMaxScaler",
        sample=False,
        random_state=123,
        n_jobs=8,
    )
    pipe = pb.build()
    scaler = pipe.named_steps["scaler"]
    # так как сигнатура недоступна, автоподстановка не произошла
    assert not hasattr(scaler, "random_state") or getattr(scaler, "random_state", None) is None


class _FakeSampler:
    def __init__(self, sampling_strategy=None, random_state=None, n_jobs=None):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.n_jobs = n_jobs

def _fake_import_from_path(_path: str):
    # независимо от переданного пути возвращаем наш класс семплера
    return _FakeSampler


def test_imbpipeline_success_branch_without_real_imblearn(monkeypatch):
    """
    Покрываем ветку:
        from imblearn.pipeline import Pipeline as ImbPipeline
        return ImbPipeline(steps)
    (успешный импорт и возврат) — без установки imblearn.
    """
    # 1) Подменяем import_from_path в AutomationML.pipeline → не трогаем имblearn.*
    monkeypatch.setattr("AutomationML.pipeline.import_from_path", _fake_import_from_path, raising=True)

    # 2) Подкладываем модуль imblearn.pipeline с минимальным Pipeline-классом
    FakeImbPipeline = type("ImbPipeline", (), {
        "__init__": lambda self, steps: setattr(self, "named_steps", dict(steps))
    })
    sys.modules["imblearn.pipeline"] = types.SimpleNamespace(Pipeline=FakeImbPipeline)

    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        scaler=True,
        sample=True,
        sampler="undersample",     # alias → _create_sampler → _create_instance → import_from_path (замокан)
        sampling_strategy=0.3,
        random_state=123,
    )
    pipe = pb.build()  # должно пройти по успешной ветке и вернуть FakeImbPipeline(steps)

    assert "sampler" in pipe.named_steps
    assert isinstance(pipe.named_steps["sampler"], _FakeSampler)

    # уборка подложенного модуля
    sys.modules.pop("imblearn.pipeline", None)


def test_imbpipeline_missing_module_raises_optional_dependency(monkeypatch):
    """
    Покрываем ветку:
        except ModuleNotFoundError:
            raise OptionalDependencyError(...)
    — когда import imblearn.pipeline падает.
    """
    # 1) Семплер снова создаём без обращения к реальному imblearn
    monkeypatch.setattr("AutomationML.pipeline.import_from_path", _fake_import_from_path, raising=True)

    # 2) Ломаем именно импорт 'imblearn.pipeline'
    real_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == "imblearn.pipeline":
            raise ModuleNotFoundError("No imblearn")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import, raising=True)

    pb = PipelineBuilder(
        model="sklearn.linear_model.LogisticRegression",
        scaler=True,
        sample=True,
        sampler="undersample",
    )
    with pytest.raises(OptionalDependencyError):
        pb.build()
