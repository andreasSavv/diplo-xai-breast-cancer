"""
lrp_analysis.py

Layer-wise Relevance Propagation (LRP, Bach et al., PLOS ONE 2015) στο MLP του Άρθρου 2.

Τι κάνει το LRP: παίρνει την έξοδο του δικτύου για ένα δείγμα (εδώ το logit της κλάσης 1)
και τη μοιράζει προς τα πίσω, στρώμα προς στρώμα, μέχρι τα χαρακτηριστικά εισόδου. Κάθε
νευρώνας περνάει στους νευρώνες του προηγούμενου στρώματος ένα μερίδιο της "σημαντικότητάς"
του, ανάλογο με το πόσο συνεισέφεραν στην ενεργοποίησή του. Το αποτέλεσμα είναι ένας αριθμός
ανά χαρακτηριστικό και ανά δείγμα (θετικός = υπέρ της κλάσης 1, αρνητικός = κατά).

Εφαρμόζεται μόνο σε MLP (έχει στρώματα με γνωστά βάρη). Δεν εφαρμόζεται στα kNN, SVM,
Logistic Regression, RUSBoost, GRNN, ούτε εδώ στο TabNet και στον Autoencoder.

Δύο κανόνες:
- LRP-epsilon: ο βασικός κανόνας. Για δίκτυα ReLU ταυτίζεται με Gradient x Input
  (Ancona et al., ICLR 2018). Το επαληθεύουμε αριθμητικά παρακάτω.
- LRP-gamma (gamma = 0.25): ενισχύει τις θετικές συνεισφορές στα κρυφά στρώματα (όπου οι
  ενεργοποιήσεις είναι ReLU, άρα >= 0). Στο πρώτο στρώμα, όπου η είσοδος είναι τυποποιημένα
  χαρακτηριστικά (και αρνητικές τιμές), χρησιμοποιείται ο κανόνας epsilon. Αυτή η επιλογή είναι δική μου.

Η καθολική σημαντικότητα ενός χαρακτηριστικού είναι ο μέσος όρος των απόλυτων τιμών του
LRP πάνω στα δείγματα (όπως στο SHAP). Χρησιμοποιούνται τα ίδια δείγματα test με τα SHAP και LIME.

Χρήση (από τον φάκελο article2_deep_learning, με ενεργό το venv του):
    python lrp_analysis.py coimbra
    python lrp_analysis.py wdbc
"""

import importlib.util
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

SEED = 42
OUTPUT_DIR = "results_lrp"
COMPARE_DIR = "results_xai_all"   # εδώ βρίσκονται τα SHAP / LIME / permutation από το xai_all_models.py
SAMPLE_SIZE = 40                  # ίδιο με SHAP_TEST_SAMPLES / LIME_TEST_SAMPLES
GAMMA = 0.25
EPS = 1e-6

CONFIG = {
    "coimbra": {"mlp": (20,), "mlp_deep": (20, 10)},
    "wdbc": {"mlp": (10,), "mlp_deep": (10, 5)},
}


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


# ---------- Forward πέρασμα του sklearn MLP (relu στα κρυφά, γραμμική έξοδος = logit) ----------

def forward(model, X):
    acts, zs = [np.asarray(X, dtype=np.float64)], []
    a = acts[0]
    last = len(model.coefs_) - 1
    for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
        z = a @ W + b
        zs.append(z)
        a = np.maximum(z, 0.0) if i < last else z
        acts.append(a)
    return acts, zs


# ---------- LRP ----------

def lrp(model, X, rule="epsilon", gamma=GAMMA, eps=EPS):
    """Επιστρέφει πίνακα (δείγματα x χαρακτηριστικά) με τη σημαντικότητα LRP για το logit της κλάσης 1."""
    acts, zs = forward(model, X)
    R = zs[-1].copy()                       # σημαντικότητα στην έξοδο = logit
    for l in range(len(model.coefs_) - 1, -1, -1):
        W, b, a = model.coefs_[l], model.intercepts_[l], acts[l]
        if rule == "gamma" and l > 0:
            W = W + gamma * np.maximum(W, 0.0)
        z = a @ W + b
        z = z + eps * np.where(z >= 0, 1.0, -1.0)
        R = a * ((R / z) @ W.T)
    return R


def gradient_x_input(model, X):
    """Gradient x Input για το logit, υπολογισμένο με το χέρι (backprop στο numpy)."""
    acts, zs = forward(model, X)
    g = np.ones((len(acts[0]), 1))
    for l in range(len(model.coefs_) - 1, -1, -1):
        g = g @ model.coefs_[l].T
        if l > 0:
            g = g * (zs[l - 1] > 0)
    return g * acts[0]


# ---------- Βοηθητικά ----------

def top(series, k=5):
    return list(series.sort_values(ascending=False).head(k).index)


def importance(R, feature_names):
    return pd.Series(np.abs(R).mean(axis=0), index=feature_names)


def agreement(a, b):
    return spearmanr(a.values, b.values).correlation, len(set(top(a)) & set(top(b)))


def load_compare(dataset):
    """Φορτώνει SHAP / LIME / permutation του MLP, αν υπάρχουν από το xai_all_models.py."""
    out = {}
    for label, fname in [("SHAP", "shap"), ("LIME", "lime"), ("Permutation", "permutation")]:
        path = os.path.join(COMPARE_DIR, f"{fname}_{dataset}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            if "MLP" in df.columns:
                out[label] = df["MLP"]
    return out


def bar_compare(table, dataset, path, k=10):
    norm = table.clip(lower=0)
    norm = norm / norm.sum(axis=0)
    order = norm.mean(axis=1).sort_values(ascending=False).head(k).index[::-1]
    sub = norm.loc[order]
    fig, ax = plt.subplots(figsize=(9, 0.55 * k + 2))
    h = 0.8 / len(sub.columns)
    for i, col in enumerate(sub.columns):
        ax.barh(np.arange(len(sub)) + i * h, sub[col].values, height=h, label=col)
    ax.set_yticks(np.arange(len(sub)) + 0.4 - h / 2)
    ax.set_yticklabels(sub.index, fontsize=8)
    ax.set_xlabel("σημαντικότητα (κανονικοποιημένη ώστε κάθε τεχνική να αθροίζει στο 1)")
    ax.set_title(f"MLP: LRP σε σύγκριση με SHAP, LIME και permutation ({dataset})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bar_sample(R, names, title, path, k=10):
    s = pd.Series(R, index=names)
    s = s.reindex(s.abs().sort_values(ascending=False).head(k).index)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.45 * k + 1.8))
    ax.barh(s.index, s.values, color=["#c0392b" if v < 0 else "#2874a6" for v in s.values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("σημαντικότητα LRP (μπλε = υπέρ της κλάσης 1, κόκκινο = κατά)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------- Κύριο πείραμα ----------

def run(dataset):
    data = load_data(dataset)
    cfg = CONFIG[dataset]
    names = data["feature_names"]
    X_tr, y_tr = data["X_train"].values, data["y_train"].values
    X_te, y_te = data["X_test"].values, data["y_test"].values
    sample = data["X_test"].sample(n=min(SAMPLE_SIZE, len(data["X_test"])), random_state=SEED)
    X_s = sample.values
    print(f"Dataset: {dataset} | train={len(X_tr)}, test={len(X_te)}, "
          f"δείγματα για LRP={len(X_s)}, χαρακτηριστικά={len(names)}")

    models = {}
    for label, layers in [("MLP", cfg["mlp"]), (f"MLP{str(cfg['mlp_deep']).replace(' ', '')}", cfg["mlp_deep"])]:
        models[label] = MLPClassifier(hidden_layer_sizes=layers, activation="relu", solver="adam",
                                      max_iter=2000, random_state=SEED).fit(X_tr, y_tr)

    print("\nΈλεγχος επανεκπαίδευσης (πρέπει να ταιριάζει με τα προηγούμενα αποτελέσματα):")
    for label, m in models.items():
        p = m.predict_proba(X_te)[:, 1]
        print(f"  {label:14s} Accuracy={accuracy_score(y_te, (p >= 0.5).astype(int)):.4f}  "
              f"ROC-AUC={roc_auc_score(y_te, p):.4f}")

    print("\nΈλεγχοι ορθότητας της υλοποίησης:")
    for label, m in models.items():
        acts, zs = forward(m, X_s)
        logit = zs[-1][:, 0]
        p_manual = 1 / (1 + np.exp(-logit))
        d1 = np.abs(p_manual - m.predict_proba(X_s)[:, 1]).max()
        r_eps = lrp(m, X_s, "epsilon", eps=1e-9)
        d2 = np.abs(r_eps - gradient_x_input(m, X_s)).max()
        print(f"  {label:14s} διαφορά με predict_proba = {d1:.2e} | "
              f"διαφορά LRP-epsilon με Gradient x Input = {d2:.2e}  (και τα δύο πρέπει να είναι ~0)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    compare = load_compare(dataset)
    if not compare:
        print(f"\nΠΡΟΣΟΧΗ: δεν βρέθηκαν τα SHAP/LIME/permutation στον φάκελο {COMPARE_DIR}/. "
              f"Η σύγκριση παραλείπεται.")

    results = {}
    for label, m in models.items():
        r_eps = lrp(m, X_s, "epsilon")
        r_gam = lrp(m, X_s, "gamma")
        imp = pd.DataFrame({"LRP-epsilon": importance(r_eps, names), "LRP-gamma": importance(r_gam, names)})
        imp.to_csv(f"{OUTPUT_DIR}/lrp_mean_abs_{dataset}_{label}.csv")
        results[label] = imp

        print(f"\n{'=' * 78}\n{label}: TOP-5 χαρακτηριστικά ({dataset})\n{'=' * 78}")
        for col in imp.columns:
            print(f"  {col:12s}: " + ", ".join(f"{n} ({imp[col][n]:.3f})" for n in top(imp[col])))
        rho, common = agreement(imp["LRP-epsilon"], imp["LRP-gamma"])
        print(f"  Συμφωνία LRP-epsilon / LRP-gamma: Spearman {rho:.2f}, κοινά στο top-5: {common}")

        # Παράδειγμα εξήγησης ενός δείγματος
        i = 0
        p_i = m.predict_proba(X_s[i:i + 1])[0, 1]
        true_label = int(data["y_test"].loc[sample.index[i]])
        bar_sample(r_gam[i], names,
                   f"{label}, {dataset}: LRP-gamma για ένα δείγμα (P(κλάση 1)={p_i:.2f}, πραγματική κλάση={true_label})",
                   f"{OUTPUT_DIR}/lrp_sample_{dataset}_{label}.png")

    # Σύγκριση με τις άλλες τεχνικές (μόνο για το MLP με ένα κρυφό στρώμα, που είναι αυτό του xai_all_models.py)
    if compare:
        base = results["MLP"]
        table = pd.concat([base, pd.DataFrame(compare)], axis=1)
        table.to_csv(f"{OUTPUT_DIR}/lrp_vs_others_{dataset}.csv")
        print(f"\n{'=' * 78}\nMLP: συμφωνία LRP με SHAP / LIME / permutation ({dataset})  [Spearman / κοινά στο top-5]\n{'=' * 78}")
        print(f"{'':14s}" + "".join(f"{c:>16s}" for c in compare))
        for col in ["LRP-epsilon", "LRP-gamma"]:
            cells = []
            for other in compare:
                rho, common = agreement(table[col], table[other])
                cells.append(f"{rho:5.2f} / {common}")
            print(f"{col:14s}" + "".join(f"{c:>16s}" for c in cells))
        for other in compare:
            print(f"  {other:12s} top-5: " + ", ".join(top(table[other])))
        bar_compare(table, dataset, f"{OUTPUT_DIR}/lrp_vs_others_{dataset}.png")

    print(f"\nΤα CSV και τα γραφήματα αποθηκεύτηκαν στον φάκελο {OUTPUT_DIR}/")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "coimbra"
    if which not in CONFIG:
        sys.exit("Χρήση: python lrp_analysis.py [coimbra|wdbc]")
    run(which)
