"""
before_after_analysis.py

Εφαρμογή τριών τεχνικών επεξηγησιμότητας στα μοντέλα του Άρθρου 2 ΠΡΙΝ και ΜΕΤΑ την
εκπαίδευση, όπως ζήτησε ο επιβλέπων:

1. Attention visualisation: οι μάσκες προσοχής (attention masks) του TabNet.
   Μόνο το TabNet έχει attention, γι' αυτό εφαρμόζεται μόνο εκεί.
2. LRP (Layer-wise Relevance Propagation): στα MLP, που έχουν στρώματα με γνωστά βάρη.
3. CKA (Centered Kernel Alignment): στα στρώματα των MLP, του Autoencoder και του TabNet.

"Πριν την εκπαίδευση" = το ίδιο μοντέλο, ίδια αρχιτεκτονική και ίδιο seed, με τα αρχικά
(τυχαία) βάρη του, χωρίς να έχει δει τις ετικέτες. "Μετά" = το ίδιο μοντέλο εκπαιδευμένο
στα δεδομένα train. Τα δεδομένα στα οποία υπολογίζονται οι τεχνικές είναι τα ίδια και στις δύο
περιπτώσεις, ώστε η σύγκριση να είναι δίκαιη.

Πώς προκύπτει το μοντέλο "πριν":
- sklearn MLP / Autoencoder: εκπαίδευση 1 epoch με learning rate 1e-12, δηλαδή τα βάρη μένουν
  πρακτικά ίσα με την αρχική (τυχαία) αρχικοποίηση.
- TabNet: εκπαίδευση 1 epoch με learning rate 0, δηλαδή τα βάρη μένουν στην αρχικοποίηση
  (μόνο τα στατιστικά του batch normalization ενημερώνονται από τα δεδομένα εισόδου).

Χρήση (από τον φάκελο article2_deep_learning, με ενεργό το venv του):
    python before_after_analysis.py coimbra
    python before_after_analysis.py wdbc
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor

from autoencoder import encode
from cka_analysis import (CONFIG, SEED, AutoencoderClassifier, cka_matrix, heatmap,
                          linear_cka, load_data, logit_from_proba, mlp_layers,
                          tabnet_layers, train_tabnet)
from lrp_analysis import SAMPLE_SIZE, importance, lrp, top

warnings.filterwarnings("ignore")

OUTPUT_DIR = "results_before_after"
TINY_LR = 1e-12


# ---------- Μοντέλα "πριν την εκπαίδευση" ----------

def untrained_mlp(hidden, X, y, seed=SEED):
    return MLPClassifier(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                         learning_rate_init=TINY_LR, max_iter=1, random_state=seed).fit(X, y)


def trained_mlp(hidden, X, y, seed=SEED):
    return MLPClassifier(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                         max_iter=2000, random_state=seed).fit(X, y)


def untrained_encoder(bottleneck, X, seed=SEED):
    """Autoencoder με τυχαία αρχικά βάρη. Επιστρέφει μόνο την αναπαράσταση του bottleneck."""
    ae = MLPRegressor(hidden_layer_sizes=(bottleneck,), activation="relu", solver="adam",
                      learning_rate_init=TINY_LR, max_iter=1, random_state=seed).fit(X, X)
    return encode(ae, X)


def untrained_tabnet(X, y):
    from pytorch_tabnet.tab_model import TabNetClassifier
    model = TabNetClassifier(n_d=4, n_a=4, n_steps=2, gamma=1.3, lambda_sparse=1e-4,
                             seed=SEED, verbose=0, optimizer_params=dict(lr=0.0))
    model.fit(X.astype(np.float32), y.astype(np.int64), max_epochs=1, batch_size=16)
    return model


def perf(name, p, y):
    return (f"  {name:30s} Accuracy={accuracy_score(y, (p >= 0.5).astype(int)):.4f}  "
            f"ROC-AUC={roc_auc_score(y, p):.4f}")


# ---------- 1. Attention (TabNet) ----------

def tabnet_attention(model, X):
    """Επιστρέφει (σημαντικότητα ανά χαρακτηριστικό, dict μασκών ανά βήμα, συνολική μάσκα ανά δείγμα)."""
    agg, masks = model.explain(np.asarray(X, dtype=np.float32))
    agg = np.asarray(agg)
    return agg.mean(axis=0), {k: np.asarray(v) for k, v in masks.items()}, agg


def plot_attention(feature_names, y, agg_before, agg_after, path, title):
    order = np.argsort(y, kind="stable")           # πρώτα τα δείγματα κλάσης 0, μετά της κλάσης 1
    split = int((np.asarray(y) == 0).sum())
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    vmax = max(agg_before.max(), agg_after.max())
    for ax, mat, label in ((axes[0], agg_before, "Πριν την εκπαίδευση"),
                           (axes[1], agg_after, "Μετά την εκπαίδευση")):
        im = ax.imshow(mat[order], aspect="auto", cmap="magma", vmin=0, vmax=vmax)
        ax.axhline(split - 0.5, color="white", lw=1)
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=90, fontsize=7)
        ax.set_ylabel("δείγματα test (πάνω: κλάση 0, κάτω: κλάση 1)")
        ax.set_title(label, fontsize=10)
    fig.colorbar(im, ax=axes[:2], label="attention (συνολική μάσκα)", shrink=0.8)
    mb, ma = agg_before.mean(axis=0), agg_after.mean(axis=0)
    idx = np.argsort(ma)[::-1][:12][::-1]
    pos = np.arange(len(idx))
    axes[2].barh(pos - 0.2, mb[idx], height=0.4, label="πριν", color="#9aa5b1")
    axes[2].barh(pos + 0.2, ma[idx], height=0.4, label="μετά", color="#1f77b4")
    axes[2].set_yticks(pos)
    axes[2].set_yticklabels([feature_names[i] for i in idx], fontsize=8)
    axes[2].set_title("Μέση προσοχή, top-12 μετά την εκπαίδευση", fontsize=10)
    axes[2].legend()
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- 2. LRP ----------

def plot_lrp(before, after, path, title, k=10):
    idx = after.sort_values(ascending=False).head(k).index[::-1]
    pos = np.arange(len(idx))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(pos - 0.2, before[idx].values, height=0.4, label="πριν", color="#9aa5b1")
    ax.barh(pos + 0.2, after[idx].values, height=0.4, label="μετά", color="#1f77b4")
    ax.set_yticks(pos)
    ax.set_yticklabels(idx, fontsize=8)
    ax.set_xlabel("μέση |σημαντικότητα LRP-ε|")
    ax.set_title(title, fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------- Κύριο πείραμα ----------

def run(dataset):
    data = load_data(dataset)
    cfg = CONFIG[dataset]
    names = list(data["feature_names"])
    X_tr, y_tr = data["X_train"].values, data["y_train"].values
    X_te, y_te = data["X_test"].values, data["y_test"].values
    X_all = np.vstack([X_tr, X_te])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Dataset: {dataset} | train={len(X_tr)}, test={len(X_te)}, "
          f"χαρακτηριστικά={len(names)}")

    # ---- Μοντέλα πριν / μετά ----
    mlps = {"MLP": cfg["mlp"], f"MLP{str(cfg['mlp_deep']).replace(' ', '')}": cfg["mlp_deep"]}
    before, after = {}, {}
    for label, hidden in mlps.items():
        before[label] = untrained_mlp(hidden, X_tr, y_tr)
        after[label] = trained_mlp(hidden, X_tr, y_tr)

    print("\nΑπόδοση στο test, πριν και μετά (έλεγχος ότι το 'πριν' είναι πράγματι μη εκπαιδευμένο):")
    for label in mlps:
        print(perf(f"{label} πριν", before[label].predict_proba(X_te)[:, 1], y_te))
        print(perf(f"{label} μετά", after[label].predict_proba(X_te)[:, 1], y_te))

    tn_before = tn_after = None
    try:
        tn_before = untrained_tabnet(X_tr, y_tr)
        tn_after = train_tabnet(X_tr, y_tr)
        print(perf("TabNet πριν", tn_before.predict_proba(X_te.astype(np.float32))[:, 1], y_te))
        print(perf("TabNet μετά", tn_after.predict_proba(X_te.astype(np.float32))[:, 1], y_te))
        print("  (το Accuracy/AUC του TabNet εδώ έχει early stopping σε 20% του train και δεν αναφέρεται ως αποτέλεσμα)")
    except Exception as e:  # noqa: BLE001
        print(f"\nΠΡΟΣΟΧΗ: το TabNet παραλείπεται ({type(e).__name__}: {e}). Στείλε μου αυτό το μήνυμα.")
        tn_before = tn_after = None

    # ================= 1. ATTENTION =================
    print(f"\n{'=' * 78}\n1. Attention visualisation (TabNet) - πριν και μετά ({dataset})\n{'=' * 78}")
    if tn_before is not None:
        try:
            imp_b, masks_b, agg_b = tabnet_attention(tn_before, X_te)
            imp_a, masks_a, agg_a = tabnet_attention(tn_after, X_te)
            att = pd.DataFrame({"before": imp_b, "after": imp_a}, index=names)
            att.to_csv(f"{OUTPUT_DIR}/attention_{dataset}.csv")
            for step in masks_a:
                pd.DataFrame({"before": masks_b[step].mean(axis=0), "after": masks_a[step].mean(axis=0)},
                             index=names).to_csv(f"{OUTPUT_DIR}/attention_step{step}_{dataset}.csv")
            plot_attention(names, y_te, agg_b, agg_a, f"{OUTPUT_DIR}/attention_{dataset}.png",
                           f"TabNet attention πριν και μετά την εκπαίδευση ({dataset})")
            rho = spearmanr(att["before"], att["after"]).correlation
            print("Top-5 προσοχής ΠΡΙΝ  :", ", ".join(f"{n} ({v:.3f})" for n, v in att["before"].sort_values(ascending=False).head(5).items()))
            print("Top-5 προσοχής ΜΕΤΑ  :", ", ".join(f"{n} ({v:.3f})" for n, v in att["after"].sort_values(ascending=False).head(5).items()))
            print(f"Spearman ρ (πριν vs μετά) = {rho:.3f}")
            print(f"Εντροπία της μέσης προσοχής (χαμηλότερη = πιο εστιασμένη): "
                  f"πριν {-(imp_b / imp_b.sum() * np.log(imp_b / imp_b.sum() + 1e-12)).sum():.3f}, "
                  f"μετά {-(imp_a / imp_a.sum() * np.log(imp_a / imp_a.sum() + 1e-12)).sum():.3f} "
                  f"(ομοιόμορφη = {np.log(len(names)):.3f})")
        except Exception as e:  # noqa: BLE001
            print(f"ΠΡΟΣΟΧΗ: αποτυχία στο attention ({type(e).__name__}: {e}). Στείλε μου αυτό το μήνυμα.")
    else:
        print("(παραλείφθηκε, δεν υπάρχει TabNet)")

    # ================= 2. LRP =================
    print(f"\n{'=' * 78}\n2. LRP-epsilon στα MLP - πριν και μετά ({dataset})\n{'=' * 78}")
    X_s = data["X_test"].sample(n=min(SAMPLE_SIZE, len(data["X_test"])), random_state=SEED).values
    print(f"(μέσος όρος |LRP| στα ίδια {len(X_s)} δείγματα test που χρησιμοποιήθηκαν στα SHAP/LIME)")
    for label in mlps:
        imp_b = importance(lrp(before[label], X_s, rule="epsilon"), names)
        imp_a = importance(lrp(after[label], X_s, rule="epsilon"), names)
        pd.DataFrame({"before": imp_b, "after": imp_a}).to_csv(f"{OUTPUT_DIR}/lrp_{dataset}_{label}.csv")
        plot_lrp(imp_b, imp_a, f"{OUTPUT_DIR}/lrp_{dataset}_{label}.png",
                 f"LRP-ε πριν και μετά την εκπαίδευση - {label} ({dataset})")
        rho = spearmanr(imp_b.values, imp_a.values).correlation
        overlap = len(set(top(imp_b)) & set(top(imp_a)))
        print(f"\n[{label}]")
        print("  Top-5 ΠΡΙΝ :", ", ".join(f"{n} ({imp_b[n]:.3f})" for n in top(imp_b)))
        print("  Top-5 ΜΕΤΑ :", ", ".join(f"{n} ({imp_a[n]:.3f})" for n in top(imp_a)))
        print(f"  Spearman ρ (πριν vs μετά) = {rho:.3f} | κοινά top-5 = {overlap}/5")
        print(f"  Συνολική σημαντικότητα (άθροισμα μέσων |R|): πριν {imp_b.sum():.3f}, μετά {imp_a.sum():.3f}")

    # ================= 3. CKA =================
    print(f"\n{'=' * 78}\n3. CKA - πριν και μετά ({dataset})\n{'=' * 78}")
    print(f"(υπολογισμός πάνω σε όλα τα {len(X_all)} δείγματα, χωρίς ετικέτες)")

    reps_b, reps_a = {}, {}
    for label in mlps:
        for k, v in mlp_layers(before[label], X_all).items():
            reps_b[f"{label} {k}"] = v
        for k, v in mlp_layers(after[label], X_all).items():
            reps_a[f"{label} {k}"] = v

    ae_after = AutoencoderClassifier(cfg["bottleneck"]).fit(X_tr, y_tr)
    z_before = untrained_encoder(cfg["bottleneck"], X_all)
    key = f"AE bottleneck({cfg['bottleneck']})"
    reps_b[key] = z_before
    reps_a[key] = ae_after.layers(X_all)[f"bottleneck({cfg['bottleneck']})"]

    if tn_before is not None:
        lb, la = tabnet_layers(tn_before, X_all), tabnet_layers(tn_after, X_all)
        if lb is not None and la is not None:
            for k in la:
                reps_b[f"TabNet {k}"] = lb[k]
                reps_a[f"TabNet {k}"] = la[k]

    # 3α. CKA(πριν, μετά) ανά στρώμα + σχέση με την είσοδο
    rows = []
    for k in reps_a:
        rows.append({
            "layer": k,
            "CKA(before, after)": linear_cka(reps_b[k], reps_a[k]),
            "CKA(input, before)": linear_cka(X_all, reps_b[k]),
            "CKA(input, after)": linear_cka(X_all, reps_a[k]),
        })
    table = pd.DataFrame(rows).set_index("layer")

    # Αναφορά: δύο διαφορετικές τυχαίες αρχικοποιήσεις (πριν την εκπαίδευση)
    base = {}
    for label, hidden in mlps.items():
        other = untrained_mlp(hidden, X_tr, y_tr, seed=SEED + 1)
        layers_o = mlp_layers(other, X_all)
        for k, v in layers_o.items():
            base[f"{label} {k}"] = linear_cka(reps_b[f"{label} {k}"], v)
    base[key] = linear_cka(z_before, untrained_encoder(cfg["bottleneck"], X_all, seed=SEED + 1))
    table["CKA(init seed A, init seed B)"] = pd.Series(base)
    table.to_csv(f"{OUTPUT_DIR}/cka_before_after_{dataset}.csv")
    print("\nΠίνακας ανά στρώμα:")
    print("  CKA(before, after)  : πόσο μοιάζει το στρώμα πριν με το στρώμα μετά την εκπαίδευση")
    print("                        (χαμηλό = η εκπαίδευση άλλαξε πολύ την αναπαράσταση)")
    print("  CKA(input, before/after): πόσο μοιάζει το στρώμα με τα αρχικά δεδομένα")
    print("  CKA(init seed A, B) : αναφορά, δύο διαφορετικές τυχαίες αρχικοποιήσεις, χωρίς εκπαίδευση")
    print(table.round(3).to_string())

    # 3β. Πλήρεις πίνακες CKA (μέσα στα μοντέλα) πριν και μετά
    for tag, reps in (("before", reps_b), ("after", reps_a)):
        full = {"input": X_all, **reps}
        n, m = cka_matrix(full)
        pd.DataFrame(m, index=n, columns=n).to_csv(f"{OUTPUT_DIR}/cka_matrix_{tag}_{dataset}.csv")
        heatmap(m, n, f"Linear CKA ανάμεσα σε στρώματα - {'ΠΡΙΝ' if tag == 'before' else 'ΜΕΤΑ'} "
                      f"την εκπαίδευση ({dataset})", f"{OUTPUT_DIR}/cka_matrix_{tag}_{dataset}.png")

    print("\nCKA ανάμεσα στα ενδιάμεσα στρώματα διαφορετικών μοντέλων (χωρίς είσοδο και logit):")
    inner = [k for k in reps_a if not k.endswith("logit")]
    for tag, reps in (("ΠΡΙΝ", reps_b), ("ΜΕΤΑ", reps_a)):
        n, m = cka_matrix({k: reps[k] for k in inner})
        print(f"\n[{tag}]")
        print(pd.DataFrame(m, index=n, columns=n).round(2).to_string())

    print(f"\nΤα CSV και τα γραφήματα αποθηκεύτηκαν στον φάκελο {OUTPUT_DIR}/")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "coimbra"
    if which not in CONFIG:
        sys.exit("Χρήση: python before_after_analysis.py [coimbra|wdbc]")
    run(which)
