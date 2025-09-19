## Фабрика моделей (`AutomationML.factory`)

Единый интерфейс для создания экземпляров моделей машинного обучения **по строковому имени** без ручных импортов. Импорты выполняются лениво (только при первом создании нужной модели). Поддерживаются как стандартные классы `scikit-learn`, так и модели из опциональных пакетов (`xgboost`, `lightgbm`, `catboost`), а также пользовательские `callable`-фабрики.

### Ключевые идеи

* **Полные имена классов**: `"LinearRegression"`, `"RandomForestClassifier"`, `"SVC"`, `"XGBRegressor"` и т. д.
* **Ленивые импорты**: внешний пакет загружается только когда модель реально создаётся.
* **Строгая проверка параметров**: подсказки при опечатках (например, `n_estimator` → `n_estimators`).
* **Регистр имён не важен**: `"linearregression"` ≡ `"LinearRegression"`.
* **Управление реестром**: регистрация, удаление, справка, перечисление, сохранение/загрузка в JSON.
* **Политика `n_jobs`**: по умолчанию `1` там, где параметр поддерживается; можно переопределять в `create(...)`.

---

### Быстрый старт

```python
from AutomationML import ModelFactory

factory = ModelFactory(random_state=42)

# Линейная регрессия
lin = factory.create("LinearRegression", fit_intercept=True, n_jobs=1)

# Случайный лес (классификация)
rf = factory.create("RandomForestClassifier", n_estimators=300, n_jobs=-1)

# SVM-классификатор
svc = factory.create("SVC", kernel="rbf", C=2.0, probability=True)
```

Полиномиальная регрессия (конвейер `PolynomialFeatures -> LinearRegression`):

```python
poly = factory.create(
    "PolynomialRegression",
    degree=3,
    include_bias=False,
    linear_kwargs={"fit_intercept": True, "n_jobs": 2},
)
```

Ожидаемый тип задачи (защита от ошибок выбора модели):

```python
# Поднимет TaskMismatchError, так как LogisticRegression — классификация
factory.create("LogisticRegression", expected_task="reg")
```

---

### Поддерживаемые «из коробки» модели

Классификация: `LogisticRegression`, `SVC`, `KNeighborsClassifier`, `GaussianNB`,
`DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `BaggingClassifier`,
`CatBoostClassifier`, `XGBClassifier`, `LGBMClassifier`, `StackingClassifier`.

Регрессия: `LinearRegression`, `Ridge`, `Lasso`, `PolynomialRegression`, `SVR`,
`KNeighborsRegressor`, `DecisionTreeRegressor`, `RandomForestRegressor`,
`GradientBoostingRegressor`, `BaggingRegressor`, `CatBoostRegressor`, `XGBRegressor`, `LGBMRegressor`, `StackingRegressor`.

Полный список можно получить программно:

```python
factory.list_models()          # все
factory.list_models("clf")     # только классификация
factory.list_models("reg")     # только регрессия
```

---

### Регистрация пользовательских моделей

#### По строковому пути к классу

```python
factory.register_model(
    "ElasticNet",
    target="sklearn.linear_model.ElasticNet",
    defaults={"alpha": 0.1, "l1_ratio": 0.7, "max_iter": 10_000},
    task="reg",
    doc="Пользовательская линейная модель",
    eager_validate=True,  # сразу проверит корректность defaults по сигнатуре
)
enet = factory.create("ElasticNet", alpha=0.2)
```

#### Через `callable`-фабрику

```python
def my_svr_factory(**params):
    from sklearn.svm import SVR
    return SVR(**params)

factory.register_model(
    "MySVR",
    target=my_svr_factory,
    defaults={"kernel": "rbf", "C": 1.0},
    task="reg",
    doc="Пример фабрики",
)
svr = factory.create("MySVR", C=10.0)
```

Перерегистрация и удаление:

```python
factory.register_model("ElasticNet", target="sklearn.linear_model.ElasticNet", overwrite=True)
factory.unregister_model("ElasticNet")
```

> Встроенные записи защищены от удаления/перезаписи; их можно удалить только с `force=True` в `unregister_model`.

---

### Справка и список моделей

```python
factory.get_help("RandomForestRegressor")
# Модель: RandomForestRegressor
# Задача: reg
# Дефолты: {'random_state': 42, 'n_jobs': 1}
# Источник: sklearn.ensemble.RandomForestRegressor
# Описание: Случайный лес (регрессия).
```

```python
factory.has_model("linearREGRESSION")  # True
```

---

### Строгая проверка гиперпараметров

По умолчанию включена (`strict_validation=True` в конструкторе фабрики):

* На этапе до инициализации проверяются имена параметров по сигнатуре конструктора.
* Для **классов** после инициализации выполняется дополнительная проверка по `model.get_params(deep=False)`, чтобы отловить опечатки, которые могли «пройти» через `**kwargs` в `__init__`.
* Для **фабрик** (`callable`, возвращающих модель) пост-проверка **не применяется** — их параметры относятся к фабрике, а не к финальной модели.

Пример диагностического сообщения:

```
Не удалось создать модель 'RandomForestClassifier' из-за неверных параметров.
Недопустимые параметры: n_estimator.
Подсказки по возможным опечаткам:
  - n_estimator: возможно, имелось в виду: n_estimators
Допустимые параметры: bootstrap, ccp_alpha, class_weight, ..., n_estimators, n_jobs, ...
```

Отключение:

```python
factory = ModelFactory(strict_validation=False)
```

---

### Сохранение и загрузка реестра

Сохранить **только пользовательские записи**:

```python
factory.save_registry("automationml_registry.json", include_builtins=False)
```

Загрузить в новую фабрику:

```python
factory2 = ModelFactory()
factory2.load_registry_file("automationml_registry.json", eager_validate=True)
```

Формат файла:

```json
{
  "format_version": 1,
  "items": [
    {
      "model_name": "ElasticNet",
      "target": "sklearn.linear_model.ElasticNet",
      "defaults": {"alpha": 0.1, "l1_ratio": 0.7, "max_iter": 10000},
      "task": "reg",
      "doc": "Пользовательская линейная модель"
    }
  ]
}
```

> Сериализация `callable` возможна только если у функции есть импортируемый путь (`module.qualname`). Лямбда и вложенные функции сериализованы не будут; в режиме `strict=True` `dump` вернёт ошибку.

---

### Опциональные зависимости

Модели из `xgboost`, `lightgbm`, `catboost` доступны только при установленных пакетах. Иначе при `create(...)` будет `OptionalDependencyError` с подсказкой по установке. Основной рантайм-зависимостью является `scikit-learn`.

---

### Замечания про `n_jobs`

* В спецификациях по умолчанию `n_jobs=1` у моделей, где этот параметр поддерживается (`RandomForest*`, `Bagging*`, `LGBM*`, `XGB*`, `LinearRegression` и т. д.).
* Переопределять можно в `create(...)`: `n_jobs=-1` — использовать все ядра.

---

### Безопасность

Загрузка реестра ведёт к импортам модулей по строковым путям. Не загружать реестры из непроверенных источников.

---
---

## Конструктор пайплайнов (`AutomationML.pipeline`)

Единый интерфейс для сборки `sklearn`/`imblearn`-совместимого `Pipeline` по декларативной спецификации шагов без ручных импортов. Возвращаемый объект — готовый `Pipeline`, который можно использовать напрямую через `.fit()`/`.predict()`.

### Ключевые идеи

* **Шаги по умолчанию**: `scaler → sampler → model`. Дополнительно можно добавлять свои шаги через `extra_steps`.
* **Формат спецификации шага**:
  - экземпляр (например, `StandardScaler()`),
  - класс/фабрика (`StandardScaler`),
  - строковый путь (`"sklearn.preprocessing.StandardScaler"`),
  - словарь `{ "obj": <spec>, "params": {...} }`.
* **Алиасы семплеров**: `"undersample"` (RandomUnderSampler), `"smote"` (SMOTE), `"combine"` (SMOTEENN).
* **Автоподстановка `random_state`**: добавляется в шаги препроцессинга и семплинга, если они поддерживают параметр.
* **Валидация параметров**: проверка по сигнатуре конструктора, с подсказками при опечатках.
* Если в пайплайне есть `sampler` → возвращается `imblearn.Pipeline`, иначе — `sklearn.Pipeline`.

---

### Быстрый старт

```python
from AutomationML.pipeline import PipelineBuilder

pb = PipelineBuilder(
    model="sklearn.linear_model.LogisticRegression",
    scaler=True,                # StandardScaler
    sample=True,                # включает шаг семплинга
    sampler="undersample",      # алиас на RandomUnderSampler
    sampling_strategy=0.3,
)
pipe = pb.build()
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

---

### Продвинутый пример

```python
from sklearn.preprocessing import StandardScaler
from AutomationML.pipeline import PipelineBuilder

pb = PipelineBuilder(
    model="sklearn.ensemble.RandomForestClassifier",
    scaler=StandardScaler(with_mean=False),
    sampler={"alias": "smote", "params": {"k_neighbors": 5, "sampling_strategy": 0.5}},
    extra_steps=[
        {"name": "selector", "obj": "sklearn.feature_selection.SelectKBest", "params": {"k": 20}},
    ],
)
pipe = pb.build()
```

---

### Поддерживаемые шаги

В `PipelineBuilder` явно предусмотрены три основных шага:

* **scaler** — масштабирование признаков (`StandardScaler`, `MinMaxScaler`, и др.).
* **sampler** — балансировка классов (`RandomUnderSampler`, `SMOTE`, `SMOTEENN`).
* **model** — обязательный финальный шаг (классификатор или регрессор).

Кроме того, через параметр **extra_steps** можно добавить в конвейер любые дополнительные шаги препроцессинга или трансформации. Это могут быть:

* **encoder** — кодирование категориальных признаков (`OneHotEncoder`, `OrdinalEncoder`, target encoding).
* **imputer** — обработка пропусков (`SimpleImputer`, `KNNImputer`, `IterativeImputer`).
* **feature_selector** — отбор признаков (`SelectKBest`, `VarianceThreshold`, `SelectFromModel`, `RFE`).
* **feature_generator** — генерация новых признаков (`PolynomialFeatures`, взаимодействия признаков).
* **dim_reduction** — понижение размерности (`PCA`, `TruncatedSVD`, `UMAP`).
* и любые другие совместимые со sklearn трансформеры.

Таким образом, базовый конструктор остаётся простым, но при этом гибко расширяется за счёт `extra_steps`.

---

### Дополнительно

* Доступ к шагам через `pipe.named_steps`.
* Поддержка `set_params`/`get_params` для всех шагов.
* Для sampler требуется установленный пакет `imbalanced-learn`.

---

### Опциональные зависимости

* Для пайплайна: `imbalanced-learn` (только при использовании sampler).
* Для моделей: `xgboost`, `lightgbm`, `catboost` (при необходимости).

---

### Безопасность

Сборка пайплайнов по строковым путям приводит к импортам модулей. Не используйте спецификации из непроверенных источников.

---
---

## Фабрика методов подбора гиперпараметров (`AutomationML.search`)

Единый интерфейс для запуска `GridSearchCV`, `RandomizedSearchCV` и Optuna (`OptunaSearchCV` или ручной режим) по декларативному описанию пространства гиперпараметров. Используется единый DSL: категории задаются списками, числовые диапазоны — кортежами `("int"|"float", low, high, {opts})`, где `opts` поддерживает `step` (дискретизация) и `log=True` (лог-шкала для `float`). Это позволяет одной и той же спецификацией запускать разные поисковые стратегии без переписывания кода.

### Ключевые идеи

* **Единый DSL для всех методов**: один `param_space` работает в `grid`, `random`, `optuna`.
* **Знакомый sklearn-API**: для `grid`/`random` возвращаются обычные `GridSearchCV`/`RandomizedSearchCV`; для `optuna` — `OptunaSearchCV` (если доступен) или ручной конфиг.
* **Автогенерация objective**: в ручном Optuna-режиме функция цели генерируется автоматически (не нужно писать `objective` вручную).
* **Строгая валидация**: проверка имён параметров конструктора с подсказками при опечатках.
* **Ленивая интеграция**: опциональные зависимости (`optuna`, `scipy`) требуются только когда действительно нужны.

---

### Быстрый старт

```python
from AutomationML.search import SearchFactory

factory = SearchFactory(random_state=42)
```

#### GridSearchCV (для диапазонов обязателен `step`)

```python
grid = factory.create(
    method="grid",
    estimator_path="sklearn.linear_model.LogisticRegression",
    estimator_kwargs={"max_iter": 10_000},
    param_space={
        "C": ("float", 1e-3, 1e2, {"step": 1.0}),
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    },
    cv=5, scoring="accuracy", n_jobs=-1,
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
```

#### RandomizedSearchCV (без `step` → распределения; с `step` → дискретные списки)

```python
rnd = factory.create(
    method="random",
    estimator_path="sklearn.ensemble.RandomForestClassifier",
    param_space={
        "n_estimators": ("int", 50, 200),
        "max_depth": ("int", 3, 15),
        "criterion": ["gini", "entropy"],
    },
    cv=5, scoring="accuracy", n_iter=20, n_jobs=-1,
)
rnd.fit(X_train, y_train)
best_model = rnd.best_estimator_
```

#### Optuna (оба варианта: интеграция и ручной режим)

```python
opt = factory.create(
    method="optuna",
    estimator_path="sklearn.svm.SVC",
    param_space={
        "C": ("float", 1e-3, 1e3, {"log": True}),
        "kernel": ["linear", "rbf"],
        "gamma": ("float", 1e-4, 1.0, {"log": True}),
    },
    cv=5, scoring="accuracy", n_trials=40, n_jobs=-1,
)

# Если установлен optuna.integration.OptunaSearchCV — вернётся sklearn-совместимый объект:
if hasattr(opt, "fit"):
    opt.fit(X_train, y_train)
    best_model = opt.best_estimator_

# Иначе фабрика вернёт кортеж ("optuna-manual", cfg): запускаем ручной режим
else:
    from AutomationML.search import run_optuna_manual
    mode, cfg = opt
    study = run_optuna_manual(
        estimator_class=cfg["estimator_class"],
        grid=cfg["param_space"],
        X_train=X_train, y_train=y_train,
        scoring=cfg["scoring"], cv=cfg["cv"],
        n_trials=cfg["n_trials"],
        study_direction=cfg.get("study_direction", "maximize"),
        sampler=cfg.get("sampler"), pruner=cfg.get("pruner"),
        fit_params=cfg.get("fit_params"), fixed_params=cfg.get("fixed_params"),
        timeout=cfg.get("timeout"), callbacks=cfg.get("callbacks"),
    )
    # финальная модель создаётся вручную лучшими параметрами:
    from sklearn.svm import SVC
    best_model = SVC(**study.best_trial.params).fit(X_train, y_train)
```

---

### Единый DSL пространства гиперпараметров

```python
param_space = {
  "criterion": ["gini", "entropy", "log_loss"],       # категории
  "max_depth": ("int", 3, 30),                        # диапазон int (включительно)
  "min_samples_leaf": ("int", 1, 20, {"step": 3}),    # дискретизация шагом 3
  "alpha": ("float", 1e-4, 1.0, {"log": True}),       # лог-шкала (float)
  # условные/произвольные параметры только для Optuna:
  "leaf_size": lambda t: t.suggest_int("leaf_size", 10, 100, step=5),
}
```

* **GridSearchCV**: для диапазонов обязателен `step`; иначе будет ошибка валидации.
* **RandomizedSearchCV**: без `step` диапазоны превращаются в распределения (`scipy.stats`), с `step` — в дискретные списки.
* **Optuna**: диапазоны и категории транслируются в `trial.suggest_*`, `callable` исполняются как есть.

---

### Ручная Optuna (классическая схема)

Можно обойтись без `OptunaSearchCV` и без фабрики поиска:

```python
from AutomationML.search import gen_objective, run_optuna_manual
from sklearn.linear_model import Ridge

param_space = {"alpha": ("float", 1e-6, 1e-1, {"log": True})}

# автоматическая генерация objective(trial) под заданный класс модели и DSL
objective = gen_objective(
    estimator_class=Ridge,
    grid=param_space,
    X_train=X_train, y_train=y_train,
    scoring="neg_mean_squared_error",
    cv=5,
)

import optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# то же в «одну строчку»:
study = run_optuna_manual(
    Ridge, param_space, X_train, y_train,
    scoring="neg_mean_squared_error", cv=5, n_trials=30
)
```

---

### Возвращаемые объекты

* `method="grid"` → `GridSearchCV` (`.fit()`, `.best_params_`, `.best_estimator_`).
* `method="random"` → `RandomizedSearchCV` (аналогично).
* `method="optuna"`:

  * при наличии интеграции → `OptunaSearchCV` (sklearn-API);
  * иначе → кортеж `("optuna-manual", cfg)`, который передаётся в `run_optuna_manual(...)` и возвращает `optuna.study.Study`.

---

### Примеры `param_space`

**LogisticRegression**:

```python
param_space_lr = {
    "penalty": ["l1", "l2", "elasticnet"],
    "C": ("float", 1e-3, 1e2, {"log": True}),
    "solver": ["liblinear", "lbfgs", "saga"],
    "l1_ratio": ("float", 0.0, 1.0, {"step": 0.1}),  # используется при penalty="elasticnet"
    "max_iter": [10_000],
}
```

**RandomForestClassifier**:

```python
param_space_rf = {
    "n_estimators": ("int", 50, 500),
    "criterion": ["gini", "entropy"],
    "max_depth": ("int", 5, 50),
    "min_samples_split": ("int", 2, 20),
    "min_samples_leaf": ("int", 1, 20),
    "max_features": ("float", 0.1, 0.9),
    "bootstrap": [True, False],
    "oob_score": [True, False],
    "ccp_alpha": ("float", 0.0, 0.02),
}
```

---

### Ошибки и диагностика

* `UnknownSearchMethodError` — неизвестный метод (`method`).
* `InvalidSearchSpaceError` — некорректный `param_space` для выбранного метода (например, диапазон без `step` в `grid` или `callable` в `random`).
* `OptionalDependencyError` — отсутствуют опциональные зависимости:

  * `optuna` — для методов `optuna`;
  * `scipy` — для `random` с непрерывными диапазонами без `step`.

---

### Опциональные зависимости

* **optuna** — для Optuna-методов (`OptunaSearchCV` и ручной режим).
* **scipy** — для `RandomizedSearchCV`, если используются непрерывные диапазоны без `step`.
* Основной рантайм остаётся `scikit-learn`; прочие extras не обязательны.

---

### Безопасность

Пространства гиперпараметров могут содержать `callable` (только для Optuna). Не используйте такие спецификации из непроверённых источников. Импорты классов по строковым путям выполняются лениво и с проверками.
