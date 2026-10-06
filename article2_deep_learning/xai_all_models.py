"""
xai_all_models.py

Ισότιμη σύγκριση τεχνικών XAI και στα 8 μοντέλα (4 κλασικά ML + 4 deep learning),
σε δύο datasets (Coimbra, WDBC).

Τρεις τεχνικές για ΚΑΘΕ μοντέλο, ώστε η σύγκριση να είναι ισότιμη:
1. SHAP (KernelExplainer): μέση απόλυτη επίδραση κάθε χαρακτηριστικού στην πρόβλεψη.
2. Permutation importance: πόσο πέφτει το ROC-AUC όταν ανακατεύουμε ένα χαρακτηριστικό.
3. LIME: για κάθε δείγμα φτιάχνει ένα απλό γραμμικό μοντέλο γύρω από αυτό και
   διαβάζει τους συντελεστές του. Εδώ παίρνουμε τον μέσο όρο των απόλυτων συντελεστών.

Στο τέλος υπολογίζει πόσο συμφωνούν οι τρεις τεχνικές μεταξύ τους, για κάθε μοντέλο
(συντελεστής Spearman πάνω στις κατατάξεις και πλήθος κοινών στο top-5).

Χρήση (από τον φάκελο article2_deep_learning, με ενεργό το venv του):
    pip install lime
    python xai_all_models.py coimbra
    python xai_all_models.py wdbc

Τα μοντέλα είναι αυτά του πρώτου χωρισμού train/test (ίδιες ρυθμίσεις με το xai_dl.py).
"""

import importlib.util
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC
from imblearn.ensemble import RUSBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from grnn import GRNN
from autoencoder import encode

warnings.filterwarnings("ignore")

SEED = 42
OUTPUT_DIR = "results_xai_all"
SHAP_BACKGROUND = 30
SHAP_TEST_SAMPLES = 40
SHAP_NSAMPLES = 1000
PERM_REPEATS = 30
LIME_TEST_SAMPLES = 40
LIME_NSAMPLES = 1000

CONFIG = {
    "coimbra": {"sigma": 2.0, "mlp_layers": (20,), "bottleneck": 7},
    "wdbc": {"sigma": 1.0, "mlp_layers": (10,), "bottleneck": 3},
}

CLASSICAL = ["Logistic Regression", "kNN", "SVM (RBF)", "RUSBoost"]
DEEP = ["GRNN", "MLP", "Autoencoder", "TabNet"]


# ---------- Δεδομένα ----------

def load_data(name):
    if name == "coimbra":
        from data_loading import load_coimbra_dataset
        return load_coimbra_dataset()
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "article2_deep_learning_wdbc", "data_loading.py")
    spec = importlib.util.spec_from_file_location("wdbc_data_loading", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_wdbc_dataset()


# ---------- Μοντέλα (όλα με predict_proba πάνω σε numpy) ----------

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


class TabNetWrapper:
    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        return self.model.predict_proba(np.asarray(X, dtype=np.float32))


def build_models(data, cfg):
    X_tr, y_tr = data["X_train"].values, data["y_train"].values
    X_te, y_te = data["X_test"].values, data["y_test"].values
    models = {}

    models["Logistic Regression"] = LogisticRegression(max_iter=5000, random_state=SEED).fit(X_tr, y_tr)
    models["kNN"] = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
    models["SVM (RBF)"] = SVC(kernel="rbf", probability=True, random_state=SEED).fit(X_tr, y_tr)
    models["RUSBoost"] = RUSBoostClassifier(n_estimators=100, random_state=SEED).fit(X_tr, y_tr)

    models["GRNN"] = GRNN(sigma=cfg["sigma"]).fit(X_tr, y_tr)
    models["MLP"] = MLPClassifier(hidden_layer_sizes=cfg["mlp_layers"], activation="relu",
                                  solver="adam", max_iter=2000, random_state=SEED).fit(X_tr, y_tr)
    models["Autoencoder"] = AutoencoderClassifier(cfg["bottleneck"]).fit(X_tr, y_tr)

    # Ίδιες ρυθμίσεις με το xai_dl.py (συμπεριλαμβανομένου του eval_set), για συνέπεια με τα προηγούμενα XAI
    tabnet = TabNetClassifier(n_d=4, n_a=4, n_steps=2, gamma=1.3, lambda_sparse=1e-4,
                              seed=SEED, verbose=0)
    tabnet.fit(X_tr.astype(np.float32), y_tr.astype(np.int64),
               eval_set=[(X_te.astype(np.float32), y_te.astype(np.int64))],
               eval_metric=["auc"], max_epochs=300, patience=60, batch_size=16)
    models["TabNet"] = TabNetWrapper(tabnet)
    return models


# ---------- Τεχνική 1: SHAP ----------

def shap_importance(model, data, name, dataset):
    X_train = data["X_train"].values
    background = shap.sample(X_train, min(SHAP_BACKGROUND, len(X_train)), random_state=SEED)
    X_sample = data["X_test"].sample(n=min(SHAP_TEST_SAMPLES, len(data["X_test"])), random_state=SEED)

    explainer = shap.KernelExplainer(model.predict_proba, background)
    values = explainer.shap_values(X_sample.values, nsamples=SHAP_NSAMPLES, silent=True)
    if isinstance(values, list):
        values = values[1]
    elif values.ndim == 3:
        values = values[:, :, 1]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure()
    shap.summary_plot(values, X_sample, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance - {name} ({dataset})")
    plt.tight_layout()
    safe = name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"{OUTPUT_DIR}/shap_bar_{dataset}_{safe}.png", dpi=150)
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
    return pd.Series(drops, index=data["feature_names"])


# ---------- Τεχνική 3: LIME ----------

def lime_importance(model, data):
    X_train = data["X_train"].values
    n_features = X_train.shape[1]
    explainer = LimeTabularExplainer(
        X_train, feature_names=data["feature_names"], class_names=["0", "1"],
        mode="classification", discretize_continuous=False, random_state=SEED,
    )
    X_sample = data["X_test"].sample(n=min(LIME_TEST_SAMPLES, len(data["X_test"])), random_state=SEED).values

    total = np.zeros(n_features)
    for x in X_sample:
        exp = explainer.explain_instance(x, model.predict_proba, num_features=n_features,
                                         num_samples=LIME_NSAMPLES)
        for idx, weight in exp.as_map()[1]:
            total[idx] += abs(weight)
    return pd.Series(total / len(X_sample), index=data["feature_names"])


# ---------- Σύγκριση τεχνικών ----------

def top(series, k=5):
    return list(series.sort_values(ascending=False).head(k).index)


def agreement(shap_s, perm_s, lime_s):
    """Spearman πάνω στις τιμές σημαντικότητας και πλήθος κοινών στο top-5, για κάθε ζεύγος τεχνικών."""
    pairs = {"SHAP-Permutation": (shap_s, perm_s), "SHAP-LIME": (shap_s, lime_s),
             "Permutation-LIME": (perm_s, lime_s)}
    out = {}
    for label, (a, b) in pairs.items():
        rho = spearmanr(a.values, b.values).correlation
        common = len(set(top(a)) & set(top(b)))
        out[label] = (rho, common)
    return out


def run(dataset):
    data = load_data(dataset)
    cfg = CONFIG[dataset]
    print(f"Dataset: {dataset} | train={len(data['X_train'])}, test={len(data['X_test'])}, "
          f"χαρακτηριστικά={len(data['feature_names'])}")
    print("Εκπαίδευση των 8 μοντέλων (το TabNet παίρνει λίγο χρόνο)...\n")
    models = build_models(data, cfg)

    print("Έλεγχος επανεκπαίδευσης (σύγκρινε με τα προηγούμενα αποτελέσματα του πρώτου χωρισμού):")
    for name, m in models.items():
        proba = m.predict_proba(data["X_test"].values)[:, 1]
        acc = accuracy_score(data["y_test"], (proba >= 0.5).astype(int))
        auc = roc_auc_score(data["y_test"], proba)
        print(f"  {name:20s} Accuracy={acc:.4f}  ROC-AUC={auc:.4f}")

    shap_t, perm_t, lime_t = {}, {}, {}
    for name, m in models.items():
        print(f"\nΥπολογισμός SHAP + permutation + LIME για: {name}")
        shap_t[name] = shap_importance(m, data, name, dataset)
        perm_t[name] = permutation_importance(m, data)
        lime_t[name] = lime_importance(m, data)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(shap_t).to_csv(f"{OUTPUT_DIR}/shap_{dataset}.csv")
    pd.DataFrame(perm_t).to_csv(f"{OUTPUT_DIR}/permutation_{dataset}.csv")
    pd.DataFrame(lime_t).to_csv(f"{OUTPUT_DIR}/lime_{dataset}.csv")

    print(f"\n{'=' * 78}\nTOP-5 χαρακτηριστικά ανά τεχνική και μοντέλο ({dataset})\n{'=' * 78}")
    rows = []
    for name in CLASSICAL + DEEP:
        print(f"\n[{name}]")
        print("  SHAP        :", ", ".join(top(shap_t[name])))
        print("  Permutation :", ", ".join(top(perm_t[name])))
        print("  LIME        :", ", ".join(top(lime_t[name])))
        ag = agreement(shap_t[name], perm_t[name], lime_t[name])
        for label, (rho, common) in ag.items():
            rows.append({"model": name, "pair": label, "spearman": rho, "common_top5": common})

    ag_df = pd.DataFrame(rows)
    ag_df.to_csv(f"{OUTPUT_DIR}/agreement_{dataset}.csv", index=False)

    print(f"\n{'=' * 78}\nΣΥΜΦΩΝΙΑ ΤΕΧΝΙΚΩΝ ({dataset}): Spearman / κοινά στο top-5\n{'=' * 78}")
    print(f"{'Μοντέλο':22s}{'SHAP-Perm':>16s}{'SHAP-LIME':>16s}{'Perm-LIME':>16s}")
    for name in CLASSICAL + DEEP:
        cells = []
        for label in ("SHAP-Permutation", "SHAP-LIME", "Permutation-LIME"):
            r = ag_df[(ag_df.model == name) & (ag_df.pair == label)].iloc[0]
            cells.append(f"{r.spearman:5.2f} / {int(r.common_top5)}")
        print(f"{name:22s}{cells[0]:>16s}{cells[1]:>16s}{cells[2]:>16s}")

    print(f"\nΤα CSV και τα γραφήματα αποθηκεύτηκαν στον φάκελο {OUTPUT_DIR}/")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "coimbra"
    if which not in CONFIG:
        sys.exit("Χρήση: python xai_all_models.py [coimbra|wdbc]")
    run(which)
