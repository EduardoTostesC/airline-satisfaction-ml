"""Adapters mínimos: categorias aprendidas apenas no treino de cada fold."""

from numbers import Integral, Real

import numpy as np
import pandas as pd


def categorical_strings(series):
    def canonical(value):
        if pd.isna(value):
            return "M:"
        if isinstance(value, Integral):
            return "V:" + str(int(value))
        if isinstance(value, Real) and float(value).is_integer():
            return "V:" + str(int(value))
        return "V:" + str(value)

    # A codificação é a mesma; o cache reutiliza strings em vez de criar uma por linha.
    mapping = {value: canonical(value) for value in series.dropna().unique()}
    return series.astype(object).map(mapping).fillna("M:")


def prepare_fold(train, valid, test, categorical, model_name):
    frames = [frame.copy() for frame in (train, valid, test)]
    for column in categorical:
        strings = [categorical_strings(frame[column]) for frame in frames]
        if model_name == "catboost":
            for frame, values in zip(frames, strings):
                frame[column] = values
        else:
            dtype = pd.CategoricalDtype(categories=sorted(strings[0].unique()))
            for frame, values in zip(frames, strings):
                frame[column] = values.astype(dtype)
    # Nomes internos evitam restrições de caracteres dos backends.
    for frame in frames:
        frame.columns = [f"f{index}" for index in range(len(frame.columns))]
    return frames


def fit_fold(train, y_train, valid, y_valid, categorical_indices, config, path):
    name, params = config["model"], dict(config["params"])
    seed, stopping = config["seed"], config["early_stopping_rounds"]
    if name == "catboost":
        from catboost import CatBoostClassifier

        if params.get("eval_metric", "AUC") != "AUC":
            raise ValueError("CatBoost deve usar eval_metric=AUC neste pipeline.")
        if params.get("loss_function", "Logloss") != "Logloss":
            raise ValueError("CatBoost deve usar loss_function=Logloss.")
        model = CatBoostClassifier(
            **params,
            random_seed=seed,
            cat_features=categorical_indices,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(
            train,
            y_train,
            eval_set=(valid, y_valid),
            early_stopping_rounds=stopping,
            use_best_model=True,
            verbose=100,
        )
        model.save_model(str(path.with_suffix(".cbm")))
        best = int(model.tree_count_)
    elif name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, early_stopping, log_evaluation
        except ImportError as exc:
            raise ValueError("Instale as dependências opcionais .[boosting].") from exc
        if params.get("metric", "auc") != "auc" or params.get("objective", "binary") != "binary":
            raise ValueError("LightGBM deve usar objective=binary e metric=auc.")
        params.update(metric="auc", objective="binary")
        model = LGBMClassifier(**params, random_state=seed)
        model.fit(
            train,
            y_train,
            eval_X=valid,
            eval_y=y_valid,
            eval_metric="auc",
            callbacks=[
                early_stopping(stopping, first_metric_only=True, verbose=False),
                log_evaluation(100),
            ],
        )
        model.booster_.save_model(str(path.with_suffix(".txt")))
        best = int(model.best_iteration_)
    else:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ValueError("Instale as dependências opcionais .[boosting].") from exc
        if params.get("objective", "binary:logistic") != "binary:logistic":
            raise ValueError("XGBoost deve usar objective=binary:logistic.")
        if params.get("eval_metric", "auc") != "auc":
            raise ValueError("XGBoost deve usar eval_metric=auc.")
        params.update(objective="binary:logistic", eval_metric="auc")
        model = XGBClassifier(
            **params, random_state=seed, enable_categorical=True, early_stopping_rounds=stopping
        )
        model.fit(train, y_train, eval_set=[(valid, y_valid)], verbose=100)
        model.save_model(str(path.with_suffix(".ubj")))
        best = int(model.best_iteration + 1)
    if list(model.classes_) != [0, 1]:
        raise ValueError("Ordem das classes inesperada; probabilidade positiva não confirmada.")
    importance = np.asarray(model.feature_importances_, dtype=float)
    return model, best, importance
