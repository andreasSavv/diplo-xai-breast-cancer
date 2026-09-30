"""
explain_shap.py

Εφαρμογή SHAP πάνω στα 4 μοντέλα του Άρθρου 2, για να δούμε ποιοι βιοδείκτες
επηρεάζουν περισσότερο κάθε πρόβλεψη.

Χρησιμοποιούμε το shap.KernelExplainer, που δουλεύει με ΟΠΟΙΟΔΗΠΟΤΕ μοντέλο
(model-agnostic) — κατάλληλο εδώ γιατί έχουμε 4 διαφορετικούς τύπους μοντέλων
(γραμμικό, βασισμένο σε απόσταση, SVM, ensemble δέντρων).
"""

import shap
import matplotlib.pyplot as plt
import os



def explain_model(model, X_train, X_test, model_name, output_dir="results",
                   background_size=40, test_sample_size=24):
    os.makedirs(output_dir, exist_ok=True)

    # Μικρό "background" δείγμα για ταχύτητα (το KernelExplainer είναι αργό
    # αν του δώσουμε όλο το train set)
    background = shap.sample(X_train, min(background_size, len(X_train)), random_state=42)
    X_test_sample = X_test.sample(n=min(test_sample_size, len(X_test)), random_state=42)

    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_test_sample)

    # Για binary classification, κρατάμε τη συνεισφορά για την κλάση "patient" (1)
    if isinstance(shap_values, list):
        shap_values_plot = shap_values[1]
    else:
        shap_values_plot = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values

    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")

    plt.figure()
    shap.summary_plot(shap_values_plot, X_test_sample, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance — {model_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_bar_{safe_name}.png", dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values_plot, X_test_sample, show=False)
    plt.title(f"SHAP Summary — {model_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_beeswarm_{safe_name}.png", dpi=150)
    plt.close()

    print(f"  -> Αποθηκεύτηκαν: shap_bar_{safe_name}.png, shap_beeswarm_{safe_name}.png")


if __name__ == "__main__":
    from data_loading import load_coimbra_dataset
    from train_models import train_and_evaluate_all

    data = load_coimbra_dataset()
    trained_models, results, summary_table = train_and_evaluate_all(data, verbose=False)

    print("Μοντέλα εκπαιδεύτηκαν. Υπολογισμός SHAP explanations (μπορεί να πάρει 1-2 λεπτά)...\n")

    for name, model in trained_models.items():
        print(f"Υπολογισμός SHAP για: {name}")
        explain_model(model, data["X_train"], data["X_test"], name)

    print("\nΌλα τα SHAP plots αποθηκεύτηκαν στον φάκελο results/")