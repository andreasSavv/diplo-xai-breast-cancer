"""
xai_dl.py

Τεχνικές XAI (επεξηγησιμότητα) πάνω στα 4 deep learning μοντέλα:
GRNN, MLP, Autoencoder-based classifier, TabNet.

Δύο τεχνικές:
1. SHAP (KernelExplainer) - ίδια μέθοδος με αυτή που εφαρμόστηκε στα 4 κλασικά
   μοντέλα ML, ώστε να συγκρίνονται απευθείας.
2. Permutation importance - ανακατεύουμε τυχαία τις τιμές ενός χαρακτηριστικού
   στο test set και μετράμε πόσο πέφτει το ROC-AUC. Όσο μεγαλύτερη η πτώση,
   τόσο πιο σημαντικό το χαρακτηριστικό για το μοντέλο.

Το script τρέχει είτε στον φάκελο του Coimbra είτε στον φάκελο του WDBC:
επιλέγει μόνο του το σωστό dataset από το data_loading.py που βρίσκει δίπλα του.

Οι ρυθμίσεις των μοντέλων είναι οι "καλύτερες" που βρέθηκαν στα προηγούμενα
πειράματα (για κάθε dataset).
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from pytorch_tabnet.tab_model import TabNetClassifier

from grnn import GRNN
from autoencoder import encode

warnings.filterwarnings("ignore")

try:
    from data_loading import load_wdbc_dataset as load_data
    DATASET = "wdbc"
except ImportError:
    from data_loading import load_coimbra_dataset as load_data
    DATASET = "coimbra"

# Καλύτερες ρυθμίσεις ανά dataset (από τα προηγούμενα πειράματα)
CONFIG = {
    "coimbra": {"sigma": 2.0, "mlp_layers": (20,), "bottleneck": 7},
    "wdbc":    {"sigma": 1.0, "mlp_layers": (10,), "bottleneck": 3},
}[DATASET]

OUTPUT_DIR = "results_xai"
SHAP_BACKGROUND = 30
SHAP_TEST_SAMPLES = 40     # μέγιστο πλήθος test δειγμάτων που εξηγούμε με SHAP
SHAP_NSAMPLES = 1000       # πόσες "δοκιμές" κάνει το KernelExplainer ανά δείγμα
PERM_REPEATS = 30          # πόσες φορές ανακατεύουμε κάθε χαρακτηριστικό
SEED = 42


# ---------- Wrappers: όλα τα μοντέλα με την ίδια διεπαφή predict_proba ----------

class AutoencoderClassifier:
    """Autoencoder (MLPRegressor) + LogisticRegression πάνω στο bottleneck."""

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


class TabNetWrapper:
    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        return self.model.predict_proba(np.asarray(X, dtype=np.float32))


def build_models(data):
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    models = {}

    grnn = GRNN(sigma=CONFIG["sigma"]).fit(X_train, y_train)
    models["GRNN"] = grnn

    mlp = MLPClassifier(hidden_layer_sizes=CONFIG["mlp_layers"], activation="relu",
                        solver="adam", max_iter=2000, random_state=SEED)
    mlp.fit(X_train, y_train)
    models["MLP"] = mlp

    models["Autoencoder"] = AutoencoderClassifier(CONFIG["bottleneck"]).fit(X_train, y_train)

    # Ίδιες ρυθμίσεις με το tabnet_model.py (συμπεριλαμβανομένου του eval_set)
    tabnet = TabNetClassifier(n_d=4, n_a=4, n_steps=2, gamma=1.3,
                              lambda_sparse=1e-4, seed=SEED, verbose=0)
    tabnet.fit(
        X_train.values.astype(np.float32), y_train.values.astype(np.int64),
        eval_set=[(X_test.values.astype(np.float32), y_test.values.astype(np.int64))],
        eval_metric=["auc"], max_epochs=300, patience=60, batch_size=16,
    )
    models["TabNet"] = TabNetWrapper(tabnet)
    models["TabNet"].attention = tabnet.feature_importances_
    return models


# ---------- Τεχνική 1: SHAP ----------

def shap_importance(model, data, name):
    background = shap.sample(data["X_train"], min(SHAP_BACKGROUND, len(data["X_train"])), random_state=SEED)
    X_sample = data["X_test"].sample(n=min(SHAP_TEST_SAMPLES, len(data["X_test"])), random_state=SEED)

    explainer = shap.KernelExplainer(model.predict_proba, background)
    values = explainer.shap_values(X_sample, nsamples=SHAP_NSAMPLES, silent=True)
    if isinstance(values, list):
        values = values[1]
    elif values.ndim == 3:
        values = values[:, :, 1]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure()
    shap.summary_plot(values, X_sample, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance - {name} ({DATASET})")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_bar_{DATASET}_{name}.png", dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(values, X_sample, show=False)
    plt.title(f"SHAP Summary - {name} ({DATASET})")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_beeswarm_{DATASET}_{name}.png", dpi=150)
    plt.close()

    return pd.Series(np.abs(values).mean(axis=0), index=data["feature_names"])


# ---------- Τεχνική 2: Permutation importance ----------

def permutation_importance(model, data):
    rng = np.random.default_rng(SEED)
    X_test = data["X_test"].values
    y_test = data["y_test"].values
    base_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    drops = []
    for j in range(X_test.shape[1]):
        scores = []
        for _ in range(PERM_REPEATS):
            Xp = X_test.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            scores.append(roc_auc_score(y_test, model.predict_proba(Xp)[:, 1]))
        drops.append(base_auc - np.mean(scores))
    return pd.Series(drops, index=data["feature_names"]), base_auc


def top(series, k=5):
    return ", ".join(f"{n} ({v:.3f})" for n, v in series.sort_values(ascending=False).head(k).items())


if __name__ == "__main__":
    data = load_data()
    print(f"Dataset: {DATASET} | train={len(data['X_train'])}, test={len(data['X_test'])}, "
          f"χαρακτηριστικά={len(data['feature_names'])}")
    print("Εκπαίδευση των 4 μοντέλων (το TabNet παίρνει λίγο χρόνο)...\n")

    models = build_models(data)

    # Έλεγχος: τα επανεκπαιδευμένα μοντέλα πρέπει να δίνουν περίπου τα γνωστά αποτελέσματα
    print("Έλεγχος επανεκπαίδευσης (πρέπει να μοιάζει με τα προηγούμενα αποτελέσματα):")
    for name, m in models.items():
        proba = m.predict_proba(data["X_test"].values)[:, 1]
        acc = accuracy_score(data["y_test"], (proba >= 0.5).astype(int))
        auc = roc_auc_score(data["y_test"], proba)
        print(f"  {name:12s} Accuracy={acc:.4f}  ROC-AUC={auc:.4f}")

    shap_table, perm_table = {}, {}
    for name, m in models.items():
        print(f"\nΥπολογισμός SHAP + permutation importance για: {name}")
        shap_table[name] = shap_importance(m, data, name)
        perm_table[name], _ = permutation_importance(m, data)

    shap_df = pd.DataFrame(shap_table)
    perm_df = pd.DataFrame(perm_table)
    shap_df["TabNet attention"] = models["TabNet"].attention
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    shap_df.to_csv(f"{OUTPUT_DIR}/shap_mean_abs_{DATASET}.csv")
    perm_df.to_csv(f"{OUTPUT_DIR}/permutation_auc_drop_{DATASET}.csv")

    print(f"\n{'=' * 70}\nTOP-5 χαρακτηριστικά ανά τεχνική και μοντέλο ({DATASET})\n{'=' * 70}")
    for name in models:
        print(f"\n[{name}]")
        print("  SHAP (mean |SHAP|)      :", top(shap_df[name]))
        print("  Permutation (πτώση AUC) :", top(perm_df[name]))
    print("\n[TabNet - ενσωματωμένο attention]")
    print("  Attention               :", top(shap_df["TabNet attention"]))

    print(f"\nΤα γραφήματα και τα CSV αποθηκεύτηκαν στον φάκελο {OUTPUT_DIR}/")
