"""
mlp.py

MLP (Multi-Layer Perceptron) - το πιο "καθαρό" deep learning μοντέλο,
μέσω του scikit-learn (MLPClassifier).

Δοκιμάζουμε διαφορετικές αρχιτεκτονικές (πόσα κρυφά επίπεδα / νευρώνες)
για να δούμε ποια αποδίδει καλύτερα σε αυτό το μικρό dataset.
"""

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")  # αγνοούμε τα "δεν συνέκλινε πλήρως" warnings - φυσιολογικό σε μικρά datasets


ARCHITECTURES = {
    "MLP (1 layer, 10 neurons)": (10,),
    "MLP (1 layer, 20 neurons)": (20,),
    "MLP (2 layers, 10+5 neurons)": (10, 5),
    "MLP (2 layers, 20+10 neurons)": (20, 10),
}


if __name__ == "__main__":
    from data_loading import load_wdbc_dataset

    data = load_wdbc_dataset()

    best_result = None
    print("Δοκιμή διαφορετικών αρχιτεκτονικών MLP:\n")

    for name, layers in ARCHITECTURES.items():
        model = MLPClassifier(
            hidden_layer_sizes=layers,
            activation="relu",       # η πιο κοινή "μη-γραμμική" συνάρτηση ενεργοποίησης
            solver="adam",           # αλγόριθμος εκπαίδευσης (πολύ διαδεδομένος)
            max_iter=2000,           # πόσες φορές θα δει όλα τα δεδομένα κατά την εκπαίδευση
            random_state=42,
        )
        model.fit(data["X_train"], data["y_train"])

        y_pred = model.predict(data["X_test"])
        y_proba = model.predict_proba(data["X_test"])[:, 1]

        acc = accuracy_score(data["y_test"], y_pred)
        f1 = f1_score(data["y_test"], y_pred)
        auc = roc_auc_score(data["y_test"], y_proba)
        print(f"{name:32s} -> Accuracy={acc:.4f}  F1={f1:.4f}  ROC-AUC={auc:.4f}")

        if best_result is None or f1 > best_result["f1"]:
            best_result = {
                "name": name, "accuracy": acc, "f1": f1, "auc": auc,
                "precision": precision_score(data["y_test"], y_pred),
                "recall": recall_score(data["y_test"], y_pred),
            }

    print(f"\nΚαλύτερη αρχιτεκτονική: {best_result['name']}")
    print(f"Accuracy:  {best_result['accuracy']:.4f}")
    print(f"Precision: {best_result['precision']:.4f}")
    print(f"Recall:    {best_result['recall']:.4f}")
    print(f"F1-score:  {best_result['f1']:.4f}")
    print(f"ROC-AUC:   {best_result['auc']:.4f}")