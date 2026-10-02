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