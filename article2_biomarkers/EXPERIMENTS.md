## 29/9/2026 - Article 2 baseline models (Coimbra dataset)
- Έτρεξα: train_models.py (Logistic Regression, kNN, SVM RBF, RUSBoost)
- Dataset: UCI Breast Cancer Coimbra (id=451), 116 δείγματα, 9 βιοδείκτες
- Καλύτερο μοντέλο: kNN (Accuracy 83.3%, F1 0.833, ROC-AUC 0.850)
- Πλήρης πίνακας: βλ. τερματικό / output

# venv\Scripts\activate


# git add .
# git commit -m "Article 2: baseline models on Coimbra dataset (kNN best, F1=0.833)"
# git push



## 30/9/2026 - Article 2 SHAP explainability
- Έτρεξα: explain_shap.py (KernelExplainer πάνω στα 4 εκπαιδευμένα μοντέλα)
- Δείγμα: 24 test samples, background 40 samples (για ταχύτητα)
- Εύρημα: Glucose και Resistin είναι οι 2 πιο σημαντικοί βιοδείκτες σε ΟΛΑ τα μοντέλα
  (kNN, Logistic Regression, SVM, RUSBoost) - συνεπές αποτέλεσμα ανεξαρτήτως αλγορίθμου
- Output: 8 SHAP plots (bar + beeswarm) στον φάκελο results/

## 6/10/2026 - Article 2 baseline models: πλήρης πίνακας
- Ξανάτρεξα: train_models.py (ίδια αποτελέσματα με 29/9, αποθηκεύω τον πλήρη πίνακα)
- Coimbra, test set 24 δείγματα:

| Μοντέλο | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| kNN | 0.8333 | 0.9091 | 0.7692 | 0.8333 | 0.8497 |
| Logistic Regression | 0.7917 | 0.8333 | 0.7692 | 0.8000 | 0.7692 |
| SVM (RBF) | 0.7083 | 0.8000 | 0.6154 | 0.6957 | 0.7972 |
| RUSBoost | 0.6667 | 0.7273 | 0.6154 | 0.6667 | 0.6469 |