"""Воспроизводимый запуск экспериментов. Выбор модели — по 5-fold CV на train."""
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from ensemble_models import (
    TARGET, add_features, build_preprocessor, run_ensemble_experiment,
)
from grader_contracts.ensembles import EnsembleInput

SEED = 42
df = pd.read_csv("german.csv", sep=";")
X, y = df.drop(columns=[TARGET]), df[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
Xf_tr, Xf_te = add_features(X_tr), add_features(X_te)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)


def mk(model):
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


rf = lambda: RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=3, random_state=SEED)
gb = lambda: GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=SEED)
lr = lambda: LogisticRegression(C=0.3, max_iter=2000)

candidates = {
    "LogisticRegression (baseline)": lr(),
    "RandomForest default": RandomForestClassifier(n_estimators=300, random_state=SEED),
    "RandomForest depth=8, leaf=3": rf(),
    "ExtraTrees depth=10, leaf=3": ExtraTreesClassifier(n_estimators=500, max_depth=10, min_samples_leaf=3, random_state=SEED),
    "AdaBoost 200, lr=0.05": AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=SEED),
    "GradientBoosting d=2, lr=0.05": gb(),
    "GradientBoosting d=3, lr=0.1": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=SEED),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0, random_state=SEED),
    "MLP (32,16), alpha=1e-2": MLPClassifier((32, 16), alpha=1e-2, early_stopping=True, max_iter=500, random_state=SEED),
    "Soft Voting RF+GB+LR": VotingClassifier([("rf", rf()), ("gb", gb()), ("lr", lr())], voting="soft"),
}

rows = []
for name, est in candidates.items():
    t = time.time()
    pipe = mk(est)
    cv_auc = cross_val_score(pipe, Xf_tr, y_tr, cv=cv, scoring="roc_auc")
    pipe.fit(Xf_tr, y_tr)
    p = pipe.predict_proba(Xf_te)[:, 1]
    pred = (p >= 0.5).astype(int)
    rows.append({
        "model": name,
        "cv_roc_auc_train": round(float(cv_auc.mean()), 4),
        "test_roc_auc": round(float(roc_auc_score(y_te, p)), 4),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "precision": round(float(precision_score(y_te, pred)), 4),
        "recall": round(float(recall_score(y_te, pred)), 4),
        "f1": round(float(f1_score(y_te, pred)), 4),
        "seconds": round(time.time() - t, 1),
    })
    print(rows[-1], flush=True)

# выбор — только по CV на train
best = max(rows, key=lambda r: r["cv_roc_auc_train"])
print("BEST by CV:", best["model"])

res = run_ensemble_experiment(EnsembleInput(X_tr, y_tr, X_te, y_te, SEED))
final = {
    "random_seed": SEED, "split": "stratified 80/20, test_size=0.2",
    "selection_criterion": "mean 5-fold stratified CV ROC-AUC on train only",
    "experiments": rows,
    "selected_model": res.model_name,
    "final_metrics": {
        "roc_auc": res.roc_auc, "accuracy": res.accuracy, "precision": res.precision,
        "recall": res.recall, "f1": res.f1,
    },
}
json.dump(final, open("experiment_results.json", "w"), ensure_ascii=False, indent=2)
print(final["final_metrics"])
