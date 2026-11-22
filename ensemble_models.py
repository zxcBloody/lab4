"""Эксперимент с ансамблем: Soft Voting из Random Forest, Gradient Boosting и
логистической регрессии на German Credit.

Все преобразования и обучение используют только train-часть.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

TARGET = "Creditability"
NOMINAL = [
    "Purpose",
    "Sex_Marital_Status",
    "Payment_Status_of_Previous_Credit",
    "Most_valuable_available_asset",
    "Concurrent_Credits",
    "Type_of_apartment",
    "Guarantors",
]
NUMERIC = [
    "Duration_of_Credit_monthly",
    "Credit_Amount",
    "Age_years",
    "Account_Balance",
    "Value_Savings_Stocks",
    "Length_of_current_employment",
    "Instalment_per_cent",
    "Duration_in_Current_address",
    "No_of_Credits_at_this_Bank",
    "Occupation",
    "No_of_dependents",
    "Telephone",
    "Foreign_Worker",
]
ENGINEERED = ["Log_Credit_Amount", "Amount_per_Month", "Monthly_Burden"]


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Строковые признаки считаются построчно и не используют статистики test."""
    frame = pd.DataFrame(frame).copy()
    frame["Log_Credit_Amount"] = np.log1p(frame["Credit_Amount"])
    frame["Amount_per_Month"] = frame["Credit_Amount"] / frame["Duration_of_Credit_monthly"].clip(lower=1)
    frame["Monthly_Burden"] = frame["Amount_per_Month"] * frame["Instalment_per_cent"]
    return frame


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("nominal", OneHotEncoder(handle_unknown="ignore"), NOMINAL),
            ("numeric", StandardScaler(), NUMERIC + ENGINEERED),
        ]
    )


def build_model(seed: int = 42) -> Pipeline:
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=3, max_features="sqrt",
        random_state=seed, n_jobs=1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=2, subsample=0.8,
        random_state=seed,
    )
    lr = LogisticRegression(C=0.3, max_iter=2000)
    vote = VotingClassifier(
        [("rf", rf), ("gb", gb), ("lr", lr)], voting="soft", weights=[1, 1, 1]
    )
    return Pipeline([("prep", build_preprocessor()), ("model", vote)])


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучает soft-voting ансамбль на train и считает метрики на test."""
    x_train = add_features(data.train_features)
    x_test = add_features(data.test_features)
    y_train = np.asarray(data.train_target)
    y_test = np.asarray(data.test_target)

    model = build_model(data.random_seed)
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return EnsembleResult(
        model_name="Soft Voting: RandomForest + GradientBoosting + LogisticRegression",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, zero_division=0)),
        recall=float(recall_score(y_test, predictions, zero_division=0)),
        f1=float(f1_score(y_test, predictions, zero_division=0)),
    )
