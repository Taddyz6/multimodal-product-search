"""Original Week2 20-feature XGBoost baseline with Warm RMSE early stopping."""

from pathlib import Path
import time
import numpy as np
import pandas as pd
from ..schema import KEY, PAIR_KEY, LEXICAL_FEATURES, check_labels
from .io import write_json


def train_pointwise(train, validation, validation_labels, output, device="cpu"):
    """Feature tables must already be built with the declared lexical index.

    Validation keeps its full pool for later ranking metrics. Only its Warm rows
    supply RMSE targets here; Cold/unknown labels are never joined into fitting.
    """
    from xgboost import XGBRegressor

    check_labels(train, "train")
    check_labels(validation_labels, "validation")
    if device not in ["cpu", "cuda"]:
        raise ValueError("Choose the actual cpu or cuda device explicitly")
    if set(train.query_id) & set(validation.query_id):
        raise ValueError("Train/Validation query overlap")
    warm = validation.loc[validation.product_split.eq("warm")].merge(
        validation_labels[PAIR_KEY + ["grade"]], on=PAIR_KEY, validate="one_to_one"
    )
    if len(warm) != len(validation_labels):
        raise ValueError("Warm Validation feature/label mismatch")
    params = dict(
        objective="reg:squarederror",
        tree_method="hist",
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=10,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=10,
        reg_alpha=0,
        early_stopping_rounds=40,
        eval_metric="rmse",
        n_jobs=4,
        random_state=20261014,
        device=device,
    )
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    model = XGBRegressor(**params)
    began = time.perf_counter()
    model.fit(
        train[LEXICAL_FEATURES],
        train.grade.astype(np.float32),
        eval_set=[(warm[LEXICAL_FEATURES], warm.grade.astype(np.float32))],
        verbose=50,
    )
    best = int(model.best_iteration)
    # Save the selected prefix so native JSON reload uses exactly those trees.
    model.get_booster()[: best + 1].save_model(root / "model.xgb.json")
    selected = XGBRegressor()
    selected.load_model(root / "model.xgb.json")
    metadata = dict(
        kind="week2_xgboost",
        features=LEXICAL_FEATURES,
        params=params,
        best_iteration=best,
        seconds=time.perf_counter() - began,
        train_pairs=len(train),
        train_products=len(train[KEY].drop_duplicates()),
        training_scope="Train/Warm only",
        selection="Warm Validation RMSE",
        history=model.evals_result(),
    )
    write_json(root / "model.json", metadata)
    pd.DataFrame({"feature": LEXICAL_FEATURES, "importance": selected.feature_importances_}).to_csv(
        root / "feature_importance.csv", index=False
    )
    return selected, metadata
