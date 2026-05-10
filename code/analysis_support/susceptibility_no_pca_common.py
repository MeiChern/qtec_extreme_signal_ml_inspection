#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/susceptibility_no_pca_common.py
# Renamed package path: code/analysis_support/susceptibility_no_pca_common.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

import os

from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import Figure_reorganize_extreme_deformation_susceptibity as figsus
import figure6_susceptibility_stacked as fig6


def effective_n_jobs(requested: int) -> int:
    env_value = os.environ.get("NO_PCA_MODEL_N_JOBS")
    if env_value is None:
        return max(1, int(requested))
    return max(1, int(env_value))


def make_no_pca_pipeline(clf) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", clf),
        ]
    )


def build_domain_models_no_pca(
    *,
    pos_weight: float,
    stack_cv,
    n_jobs: int,
) -> dict[str, object]:
    jobs = effective_n_jobs(n_jobs)
    models: dict[str, object] = {
        "RF": make_no_pca_pipeline(
            RandomForestClassifier(
                n_estimators=350,
                random_state=figsus.SEED,
                n_jobs=jobs,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
            )
        ),
        "ET": make_no_pca_pipeline(
            ExtraTreesClassifier(
                n_estimators=400,
                random_state=figsus.SEED,
                n_jobs=jobs,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
            )
        ),
        "HGB": make_no_pca_pipeline(
            HistGradientBoostingClassifier(
                max_iter=350,
                learning_rate=0.05,
                max_depth=6,
                random_state=figsus.SEED,
            )
        ),
    }
    if fig6.HAVE_XGB:
        models["XGB"] = make_no_pca_pipeline(
            fig6.XGBClassifier(
                n_estimators=350,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=jobs,
                random_state=figsus.SEED,
                scale_pos_weight=pos_weight,
            )
        )

    estimators = [(name.lower(), clone(model)) for name, model in models.items()]
    models["Stack"] = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            max_iter=2500,
            random_state=figsus.SEED,
        ),
        stack_method="predict_proba",
        passthrough=False,
        cv=stack_cv,
        n_jobs=jobs,
    )
    return models


def install_no_pca_model_builder() -> None:
    figsus.build_domain_models = build_domain_models_no_pca
