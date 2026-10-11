"""
cka_analysis.py

Ανάλυση αναπαραστάσεων με CKA (Centered Kernel Alignment) στα deep learning μοντέλα
του Άρθρου 2 που έχουν στρώματα: MLP, Autoencoder-based classifier, TabNet.

Το CKA (Kornblith et al., ICML 2019) μετράει πόσο μοιάζουν δύο αναπαραστάσεις των ΙΔΙΩΝ
δειγμάτων. Δίνει έναν αριθμό από 0 (καμία ομοιότητα) έως 1 (ίδια γεωμετρία των δειγμάτων).
Δουλεύει και όταν οι δύο αναπαραστάσεις έχουν διαφορετικό πλήθος νευρώνων.
Εδώ χρησιμοποιείται το γραμμικό CKA.

Τι υπολογίζουμε:
1. Ένας μεγάλος πίνακας CKA ανάμεσα σε όλα τα "στρώματα" όλων των μοντέλων:
   είσοδος, κρυφά στρώματα του MLP (ένα και δύο στρώματα), bottleneck του Autoencoder,
   βήματα του TabNet, και η τελική έξοδος (logit) κάθε μοντέλου.
2. Έλεγχος αναφοράς: το ίδιο MLP εκπαιδευμένο με διαφορετικά seeds. Δείχνει πόσο
   μοιάζουν δύο εκπαιδεύσεις της ΙΔΙΑΣ αρχιτεκτονικής και δίνει κλίμακα για τις υπόλοιπες τιμές.

Σημείωση: το CKA υπολογίζεται πάνω σε όλα τα δείγματα του dataset (train + test), χωρίς
τις ετικέτες. Τα μοντέλα εκπαιδεύονται μόνο στα δεδομένα train. Το test ΔΕΝ χρησιμοποιείται
για early stopping του TabNet (χρησιμοποιείται ξεχωριστό 20% του train).

Χρήση (από τον φάκελο article2_deep_learning, με ενεργό το venv του):
    python cka_analysis.py coimbra
    python cka_analysis.py wdbc
"""

import importlib.util
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor

from autoencoder import encode

warnings.filterwarnings("ignore")

SEED = 42
OUTPUT_DIR = "results_cka"

CONFIG = {
    "coimbra": {"mlp": (20,), "mlp_deep": (20, 10), "bottleneck": 7},
    "wdbc": {"mlp": (10,), "mlp_deep": (10, 5), "bottleneck": 3},
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


# ---------- Γραμμικό CKA ----------

def linear_cka(X, Y):
    """Γραμμικό CKA ανάμεσα σε δύο πίνακες αναπαραστάσεων (δείγματα x νευρώνες)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    numerator = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    denominator = np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro")
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


# ---------- Αναπαραστάσεις ανά στρώμα ----------

def logit_from_proba(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def mlp_layers(model, X):
    """Ενεργοποιήσεις κάθε κρυφού στρώματος του MLP (relu) και το logit εξόδου."""
    out = {}
    a = X
    for i, (W, b) in enumerate(zip(model.coefs_[:-1], model.intercepts_[:-1]), start=1):
        a = np.maximum(a @ W + b, 0.0)
        out[f"hidden{i}({a.shape[1]})"] = a
    out["logit"] = logit_from_proba(model.predict_proba(X)[:, 1])
    return out


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

    def layers(self, X):
        z = encode(self.autoencoder, X)
        return {f"bottleneck({z.shape[1]})": z,
                "logit": logit_from_proba(self.predict_proba(X)[:, 1])}


def train_tabnet(X_tr, y_tr):
    from pytorch_tabnet.tab_model import TabNetClassifier
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=SEED
    )
    model = TabNetClassifier(n_d=4, n_a=4, n_steps=2, gamma=1.3, lambda_sparse=1e-4,
                             seed=SEED, verbose=0)
    model.fit(X_fit.astype(np.float32), y_fit.astype(np.int64),
              eval_set=[(X_val.astype(np.float32), y_val.astype(np.int64))],
              eval_metric=["auc"], max_epochs=300, patience=60, batch_size=16)
    return model


def tabnet_layers(model, X):
    """Έξοδοι κάθε βήματος attention του TabNet. Επιστρέφει None αν το API της βιβλιοθήκης διαφέρει."""
    try:
        import torch
        net = model.network
        net.eval()
        with torch.no_grad():
            xt = torch.tensor(np.asarray(X, dtype=np.float32)).to(next(net.parameters()).device)
            emb = net.embedder(xt)
            steps_output, _ = net.tabnet.encoder(emb)
        out = {}
        total = 0
        for i, s in enumerate(steps_output, start=1):
            s = s.cpu().numpy()
            out[f"step{i}({s.shape[1]})"] = s
            total = total + np.maximum(s, 0.0)
        out[f"decision({total.shape[1]})"] = total
        out["logit"] = logit_from_proba(model.predict_proba(np.asarray(X, dtype=np.float32))[:, 1])
        return out
    except Exception as e:  # noqa: BLE001
        print(f"  ΠΡΟΣΟΧΗ: δεν μπόρεσα να διαβάσω τα εσωτερικά στρώματα του TabNet ({type(e).__name__}: {e}).")
        print("  Το TabNet παραλείπεται από την ανάλυση. Στείλε μου αυτό το μήνυμα.")
        return None


# ---------- Γραφήματα ----------

def heatmap(matrix, labels, title, path, annotate=True):
    size = max(6, 0.55 * len(labels) + 3)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    if annotate:
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = matrix[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v < 0.6 else "black")
    fig.colorbar(im, ax=ax, label="CKA")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cka_matrix(reps):
    names = list(reps)
    m = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            m[i, j] = 1.0 if i == j else linear_cka(reps[a], reps[b])
    return names, m


# ---------- Κύριο πείραμα ----------

def run(dataset):
    data = load_data(dataset)
    cfg = CONFIG[dataset]
    X_tr, y_tr = data["X_train"].values, data["y_train"].values
    X_te, y_te = data["X_test"].values, data["y_test"].values
    X_all = np.vstack([X_tr, X_te])
    print(f"Dataset: {dataset} | train={len(X_tr)}, test={len(X_te)}, "
          f"δείγματα για CKA={len(X_all)}, χαρακτηριστικά={X_all.shape[1]}")

    # Έλεγχος ορθότητας του CKA: ίδια αναπαράσταση = 1, αμετάβλητο σε περιστροφή και αλλαγή κλίμακας
    rng = np.random.default_rng(SEED)
    Q, _ = np.linalg.qr(rng.normal(size=(X_all.shape[1], X_all.shape[1])))
    print("\nΈλεγχος του CKA (πρέπει να βγουν περίπου 1.00 / 1.00 / χαμηλό):")
    print(f"  CKA(X, X)             = {linear_cka(X_all, X_all):.3f}")
    print(f"  CKA(X, 3 * X @ Q)     = {linear_cka(X_all, 3 * X_all @ Q):.3f}  (περιστροφή + κλίμακα)")
    print(f"  CKA(X, τυχαίος θόρυβος) = {linear_cka(X_all, rng.normal(size=X_all.shape)):.3f}")

    print("\nΕκπαίδευση μοντέλων...")
    reps = {"input": X_all}

    mlp = MLPClassifier(hidden_layer_sizes=cfg["mlp"], activation="relu", solver="adam",
                        max_iter=2000, random_state=SEED).fit(X_tr, y_tr)
    for k, v in mlp_layers(mlp, X_all).items():
        reps[f"MLP {k}"] = v

    mlp_deep = MLPClassifier(hidden_layer_sizes=cfg["mlp_deep"], activation="relu", solver="adam",
                             max_iter=2000, random_state=SEED).fit(X_tr, y_tr)
    for k, v in mlp_layers(mlp_deep, X_all).items():
        reps[f"MLP{str(cfg['mlp_deep']).replace(' ', '')} {k}"] = v

    ae = AutoencoderClassifier(cfg["bottleneck"]).fit(X_tr, y_tr)
    for k, v in ae.layers(X_all).items():
        reps[f"AE {k}"] = v

    tabnet = train_tabnet(X_tr, y_tr)
    tn_layers = tabnet_layers(tabnet, X_all)
    if tn_layers is not None:
        for k, v in tn_layers.items():
            reps[f"TabNet {k}"] = v

    print("\nΈλεγχος επανεκπαίδευσης (Accuracy / ROC-AUC στο test):")
    checks = {"MLP": mlp, f"MLP{cfg['mlp_deep']}": mlp_deep, "Autoencoder": ae}
    for name, m in checks.items():
        p = m.predict_proba(X_te)[:, 1]
        print(f"  {name:18s} Accuracy={accuracy_score(y_te, (p >= 0.5).astype(int)):.4f}  "
              f"ROC-AUC={roc_auc_score(y_te, p):.4f}")
    p = tabnet.predict_proba(np.asarray(X_te, dtype=np.float32))[:, 1]
    print(f"  {'TabNet':18s} Accuracy={accuracy_score(y_te, (p >= 0.5).astype(int)):.4f}  "
          f"ROC-AUC={roc_auc_score(y_te, p):.4f}")

    # 1. Μεγάλος πίνακας CKA
    names, matrix = cka_matrix(reps)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame(matrix, index=names, columns=names)
    df.to_csv(f"{OUTPUT_DIR}/cka_all_layers_{dataset}.csv")
    heatmap(matrix, names, f"Linear CKA ανάμεσα σε όλα τα στρώματα ({dataset})",
            f"{OUTPUT_DIR}/cka_all_layers_{dataset}.png")

    print(f"\n{'=' * 78}\nCKA ανάμεσα σε στρώματα του ίδιου μοντέλου ({dataset})\n{'=' * 78}")
    for prefix in sorted({n.split(" ")[0] for n in names if n != "input"}):
        group = ["input"] + [n for n in names if n.startswith(prefix + " ")]
        print(f"\n[{prefix}]")
        sub = df.loc[group, group]
        print(sub.round(2).to_string())

    print(f"\n{'=' * 78}\nCKA ανάμεσα σε στρώματα ΔΙΑΦΟΡΕΤΙΚΩΝ μοντέλων ({dataset})\n{'=' * 78}")
    print("(τα στρώματα που εξετάζονται: τα κρυφά/ενδιάμεσα, χωρίς είσοδο και χωρίς logit)")
    inner = [n for n in names if n != "input" and not n.endswith("logit")]
    print(df.loc[inner, inner].round(2).to_string())

    print("\nΤελικές έξοδοι (logit) των μοντέλων:")
    logits = [n for n in names if n.endswith("logit")]
    print(df.loc[logits, logits].round(2).to_string())

    # 2. Έλεγχος αναφοράς: το ίδιο MLP με διαφορετικά seeds
    print(f"\n{'=' * 78}\nΑναφορά: ίδιο MLP{cfg['mlp']} με διαφορετικά seeds ({dataset})\n{'=' * 78}")
    seeds = [0, 1, 2, SEED]
    seed_reps = {}
    for s in seeds:
        m = MLPClassifier(hidden_layer_sizes=cfg["mlp"], activation="relu", solver="adam",
                          max_iter=2000, random_state=s).fit(X_tr, y_tr)
        layers = mlp_layers(m, X_all)
        hidden_key = [k for k in layers if k.startswith("hidden")][0]
        seed_reps[f"seed{s}"] = layers[hidden_key]
    s_names, s_matrix = cka_matrix(seed_reps)
    s_df = pd.DataFrame(s_matrix, index=s_names, columns=s_names)
    s_df.to_csv(f"{OUTPUT_DIR}/cka_mlp_seeds_{dataset}.csv")
    print(s_df.round(2).to_string())
    off = s_matrix[np.triu_indices(len(seeds), k=1)]
    print(f"\nΜέσο CKA μεταξύ διαφορετικών seeds: {off.mean():.3f} (min {off.min():.3f}, max {off.max():.3f})")
    print("Η τιμή αυτή είναι η 'βάση': ομοιότητες κοντά σε αυτήν σημαίνουν ότι τα στρώματα")
    print("μοιάζουν όσο μοιάζουν δύο εκπαιδεύσεις του ίδιου μοντέλου.")

    print(f"\nΤα CSV και τα γραφήματα αποθηκεύτηκαν στον φάκελο {OUTPUT_DIR}/")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "coimbra"
    if which not in CONFIG:
        sys.exit("Χρήση: python cka_analysis.py [coimbra|wdbc]")
    run(which)
