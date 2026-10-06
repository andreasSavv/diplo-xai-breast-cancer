"""
cv_all_models.py

5-fold cross-validation και για τα 8 μοντέλα (4 κλασικά ML + 4 deep learning),
σε δύο datasets: Coimbra (116 δείγματα) και WDBC (569 δείγματα).

Πώς δουλεύει:
- Τα δεδομένα χωρίζονται σε 5 ίσα μέρη (folds) με διατήρηση της αναλογίας κλάσεων.
  Κάθε μέρος γίνεται μία φορά test, ενώ τα άλλα 4 χρησιμοποιούνται για εκπαίδευση.
- Σε κάθε fold, το StandardScaler "μαθαίνει" μόνο από τα δεδομένα εκπαίδευσης.
- Οι ρυθμίσεις των GRNN (sigma), MLP (αρχιτεκτονική) και Autoencoder (bottleneck)
  επιλέγονται ΜΕΣΑ στα δεδομένα εκπαίδευσης του κάθε fold, με εσωτερικό
  3-fold cross-validation (κριτήριο: ROC-AUC). Το test fold δεν χρησιμοποιείται
  καθόλου για επιλογή ρυθμίσεων.
- Το TabNet κάνει early stopping σε ξεχωριστό 20% των δεδομένων εκπαίδευσης,
  όχι στο test fold.
- Τα κλασικά μοντέλα έχουν τις ίδιες ρυθμίσεις με το train_models.py (χωρίς tuning).

Χρήση:
    python cv_all_models.py            (και τα δύο datasets)
    python cv_all_models.py coimbra
    python cv_all_models.py wdbc
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
from imblearn.ensemble import RUSBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from grnn import GRNN
from autoencoder import encode

warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
OUTPUT_DIR = "results_cv"

SIGMAS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
MLP_ARCHS = [(10,), (20,), (10, 5), (20, 10)]
BOTTLENECKS = [3, 5, 7]


# ---------- Δεδομένα ----------

def load_full(name):
    """Επιστρέφει ΟΛΑ τα δείγματα (X, y) χωρίς χωρισμό train/test."""
    if name == "coimbra":
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=451)
        X = ds.data.features
        y = ds.data.targets.iloc[:, 0].map({1: 0, 2: 1})
    else:
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer(as_frame=True)
        X, y = d.data, d.target
    return X.values.astype(float), y.values.astype(int)


# ---------- Autoencoder + classifier (ίδιο με του xai_dl.py) ----------

class AutoencoderClassifier:
    def __init__(self, bottleneck):
        self.bottleneck = bottleneck

    def fit(self, X, y):
        self.autoencoder = MLPRegressor(
            hidden_layer_sizes=(self.bottleneck,), activation="relu",
            solver="adam", max_iter=3000, random_state=SEED,
        )
        self.autoencoder.fit(X, X)
        self.classifier = LogisticRegression(max_iter=5000, random_state=SEED)
        self.classifier.fit(encode(self.autoencoder, X), y)
        return self

    def predict_proba(self, X):
        return self.classifier.predict_proba(encode(self.autoencoder, np.asarray(X)))


class TabNetWithValidation:
    """TabNet με early stopping σε ξεχωριστό 20% των δεδομένων εκπαίδευσης."""

    def fit(self, X, y):
        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=SEED
        )
        self.model = TabNetClassifier(
            n_d=4, n_a=4, n_steps=2, gamma=1.3, lambda_sparse=1e-4,
            seed=SEED, verbose=0,
        )
        self.model.fit(
            X_fit.astype(np.float32), y_fit.astype(np.int64),
            eval_set=[(X_val.astype(np.float32), y_val.astype(np.int64))],
            eval_metric=["auc"], max_epochs=300, patience=60, batch_size=16,
        )
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(np.asarray(X, dtype=np.float32))


# ---------- Επιλογή ρυθμίσεων με εσωτερικό CV ----------

def select_by_inner_cv(candidates, make_model, X, y):
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    best_score, best_candidate = -1.0, None
    for c in candidates:
        scores = []
        for tr, va in inner.split(X, y):
            m = make_model(c).fit(X[tr], y[tr])
            scores.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
        score = float(np.mean(scores))
        if score > best_score:
            best_score, best_candidate = score, c
    return best_candidate


def fit_all_models(X_tr, y_tr):
    """Εκπαιδεύει και τα 8 μοντέλα στα δεδομένα εκπαίδευσης. Επιστρέφει (μοντέλα, επιλεγμένες ρυθμίσεις)."""
    models, chosen = {}, {}

    models["kNN"] = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
    models["Logistic Regression"] = LogisticRegression(max_iter=5000, random_state=SEED).fit(X_tr, y_tr)
    models["SVM (RBF)"] = SVC(kernel="rbf", probability=True, random_state=SEED).fit(X_tr, y_tr)
    models["RUSBoost"] = RUSBoostClassifier(n_estimators=100, random_state=SEED).fit(X_tr, y_tr)

    sigma = select_by_inner_cv(SIGMAS, lambda s: GRNN(sigma=s), X_tr, y_tr)
    models["GRNN"] = GRNN(sigma=sigma).fit(X_tr, y_tr)
    chosen["GRNN"] = f"sigma={sigma}"

    arch = select_by_inner_cv(
        MLP_ARCHS,
        lambda a: MLPClassifier(hidden_layer_sizes=a, activation="relu", solver="adam",
                                max_iter=2000, random_state=SEED),
        X_tr, y_tr,
    )
    models["MLP"] = MLPClassifier(hidden_layer_sizes=arch, activation="relu", solver="adam",
                                  max_iter=2000, random_state=SEED).fit(X_tr, y_tr)
    chosen["MLP"] = f"layers={arch}"

    bottleneck = select_by_inner_cv(BOTTLENECKS, lambda b: AutoencoderClassifier(b), X_tr, y_tr)
    models["Autoencoder"] = AutoencoderClassifier(bottleneck).fit(X_tr, y_tr)
    chosen["Autoencoder"] = f"bottleneck={bottleneck}"

    models["TabNet"] = TabNetWithValidation().fit(X_tr, y_tr)
    chosen["TabNet"] = "n_d=4, n_a=4, n_steps=2"
    return models, chosen


# ---------- Κύριο πείραμα ----------

def run_dataset(name):
    X, y = load_full(name)
    print(f"\n{'#' * 70}\nDataset: {name} | δείγματα={len(y)}, χαρακτηριστικά={X.shape[1]}\n{'#' * 70}")

    outer = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows, chosen_log = [], []
    t0 = time.time()

    for fold, (tr, te) in enumerate(outer.split(X, y), start=1):
        scaler = StandardScaler().fit(X[tr])
        X_tr, X_te = scaler.transform(X[tr]), scaler.transform(X[te])
        y_tr, y_te = y[tr], y[te]

        models, chosen = fit_all_models(X_tr, y_tr)
        chosen_log.append({"fold": fold, **chosen})

        for model_name, model in models.items():
            proba = model.predict_proba(X_te)[:, 1]
            pred = (proba >= 0.5).astype(int)
            rows.append({
                "dataset": name, "fold": fold, "model": model_name,
                "accuracy": accuracy_score(y_te, pred),
                "f1": f1_score(y_te, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_te, proba),
            })
        print(f"  fold {fold}/{N_FOLDS} ολοκληρώθηκε  (train={len(tr)}, test={len(te)}, "
              f"χρόνος από την αρχή: {time.time() - t0:.0f}s)")

    results = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results.to_csv(f"{OUTPUT_DIR}/cv_{name}_per_fold.csv", index=False)

    summary = results.groupby("model")[["accuracy", "f1", "roc_auc"]].agg(["mean", "std"])
    summary = summary.sort_values(("accuracy", "mean"), ascending=False)
    summary.to_csv(f"{OUTPUT_DIR}/cv_{name}_summary.csv")

    print(f"\n{'=' * 70}\n{N_FOLDS}-fold cross-validation ({name}) - μέσος όρος ± απόκλιση\n{'=' * 70}")
    print(f"{'Μοντέλο':22s}{'Accuracy':>16s}{'F1':>16s}{'ROC-AUC':>16s}")
    for model_name, r in summary.iterrows():
        cells = [f"{r[(m, 'mean')]:.3f} ± {r[(m, 'std')]:.3f}" for m in ("accuracy", "f1", "roc_auc")]
        print(f"{model_name:22s}{cells[0]:>16s}{cells[1]:>16s}{cells[2]:>16s}")

    print("\nΡυθμίσεις που επέλεξε το εσωτερικό CV σε κάθε fold:")
    print(pd.DataFrame(chosen_log).to_string(index=False))


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["coimbra", "wdbc"]
    for ds_name in wanted:
        run_dataset(ds_name)
    print(f"\nΤα αποτελέσματα αποθηκεύτηκαν στον φάκελο {OUTPUT_DIR}/")
