# EXPERIMENTS - Article 2 Deep Learning models

## 1/10/2026 - GRNN
- Έτρεξα: grnn.py (custom υλοποίηση, δοκιμή sigma: 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
- Καλύτερο: sigma=2.0 -> Accuracy 66.7%, F1 0.733, ROC-AUC 0.783
- Σύγκριση: χειρότερο από kNN (83.3%) και Logistic Regression (79.2%),
  καλύτερο ROC-AUC από RUSBoost

  
## 2/10/2026 - MLP
- Έτρεξα: mlp.py (sklearn MLPClassifier, δοκιμή 4 αρχιτεκτονικών)
- Καλύτερη: 1 κρυφό επίπεδο, 20 νευρώνες -> Accuracy 75.0%, F1 0.750, ROC-AUC 0.818
- Σύγκριση: 3ο καλύτερο από τα μοντέλα μέχρι τώρα (μετά kNN, Logistic Regression)
- Παρατήρηση: το πιο "deep" δίκτυο (2 επίπεδα) δεν ήταν καλύτερο - πιθανό
  overfitting λόγω μικρού dataset (92 train samples)

  
## 2/10/2026 - Autoencoder-based classifier
- Έτρεξα: autoencoder.py (MLPRegressor ως autoencoder + LogisticRegression classifier
  πάνω στο bottleneck, δοκιμή bottleneck size: 3, 5, 7)
- Καλύτερο: bottleneck=7 -> Accuracy 66.7%, F1 0.692, ROC-AUC 0.748
  (reconstruction MSE 0.207 - το χαμηλότερο, δηλαδή η καλύτερη ανακατασκευή)
- Παρατήρηση: χειρότερα από τα απλούστερα μοντέλα (kNN, LogReg, MLP) - λογικό,
  καθώς η τεχνική συμπίεσης αποδίδει συνήθως καλύτερα σε datasets με πολλά
  περισσότερα χαρακτηριστικά (π.χ. γονιδιακά δεδομένα), όχι σε 9 βιοδείκτες

  ## TabNet (Deep Learning, built-in attention-based interpretability)

Ρυθμίσεις: n_d=4, n_a=4, n_steps=2, batch_size=16, patience=60
(μικρή χωρητικότητα, προσαρμοσμένη για μικρό dataset)

Αποτελέσματα (test set, 24 δείγματα):
- Accuracy: 0.750
- Precision: 0.706
- Recall: 0.923
- F1-score: 0.800
- ROC-AUC: 0.860
- Best epoch: 22 (early stopping at epoch 82)

Ενσωματωμένη σημαντικότητα χαρακτηριστικών (attention-based):
1. Resistin (0.296)
2. Glucose (0.274)
3. BMI (0.176)
4. Age (0.099)
5. Adiponectin (0.048)
...

Σχόλιο: Το καλύτερο ROC-AUC απ' όλα τα μοντέλα (ML + DL) μέχρι στιγμής.
Σημαντική επιβεβαίωση: τα 2 πιο σημαντικά χαρακτηριστικά που εντόπισε
μόνο του το TabNet (Resistin, Glucose) ταυτίζονται με τα 2 πιο σημαντικά
χαρακτηριστικά που είχε βρει το SHAP στα 4 κλασικά μοντέλα ML — ισχυρή
ένδειξη ότι αυτοί οι δύο βιοδείκτες είναι πραγματικά κλινικά σημαντικοί,
ανεξάρτητα από τη μέθοδο εξήγησης που χρησιμοποιείται.

Με αυτό ολοκληρώθηκαν και τα 4 deep learning μοντέλα (GRNN, MLP,
Autoencoder, TabNet) που ζήτησε ο επιβλέποντας στη συνάντηση 01/10/2026.

## 6/10 - XAI στα 4 deep learning μοντέλα (xai_dl.py)

Τεχνικές: SHAP KernelExplainer (background 30 δείγματα), permutation importance (30 επαναλήψεις, πτώση ROC-AUC), attention του TabNet.
Αποτελέσματα στον φάκελο results_xai/ (PNG και CSV). Τα μοντέλα είναι του πρώτου χωρισμού.

Coimbra:
- Glucose και BMI στο top-5 του SHAP και των 4 μοντέλων. Resistin σε 3 από τα 4.
- Το permutation βάζει την Age στο top-5 και των 4 μοντέλων, ενώ το SHAP την βάζει 4η-6η. Οι τεχνικές διαφωνούν.
- Insulin, HOMA και Glucose είναι συσχετισμένα (το HOMA υπολογίζεται από τα δύο άλλα), οπότε η σειρά τους διαβάζεται με επιφύλαξη.
- Το αποτέλεσμα του BMI (υψηλές τιμές προς "υγιής") δεν συμφωνεί με την κλινική προσδοκία. Πιθανώς χαρακτηρίζει το μικρό δείγμα, δεν το γενικεύω.

WDBC:
- Κυριαρχούν χαρακτηριστικά μεγέθους (radius, perimeter, area), concave points και texture. Μόνο το worst perimeter είναι στο top-5 του SHAP και των 4 μοντέλων.
- Το radius/perimeter/area είναι γεωμετρικά συνδεδεμένα: μετράει η ομάδα, όχι ποιο ακριβώς.
- Το attention του TabNet διαφωνεί με το SHAP και το permutation (perimeter error 0.318 αλλά εκτός top-5 των άλλων τεχνικών. Το worst concave points είναι 2ο σε SHAP και permutation αλλά έχει attention 0).

Περιορισμοί: permutation στο ίδιο test set, χωρίς μέτρηση διακύμανσης. SHAP στο WDBC σε 40 από τα 114 δείγματα. Δείχνουν σε τι βασίζεται το μοντέλο, όχι αιτιότητα.


## 6/10 - 5-fold cross-validation, 8 μοντέλα, 2 datasets (cv_all_models.py)

Στρατοποιημένο 5-fold (random_state=42). Scaler ανά fold. GRNN sigma, MLP αρχιτεκτονική και Autoencoder bottleneck επιλέγονται με εσωτερικό 3-fold CV (ROC-AUC) μέσα στα δεδομένα εκπαίδευσης. Το TabNet κάνει early stopping σε 20% των δεδομένων εκπαίδευσης. Τα κλασικά μοντέλα χωρίς tuning.
Αποτελέσματα: results_cv/ (per_fold και summary CSV).

Coimbra (116 δείγματα): Accuracy μέσος ± απόκλιση
kNN 0.784 ± 0.081 | SVM 0.750 ± 0.085 | Autoencoder 0.742 ± 0.051 | MLP 0.707 ± 0.097 | Logistic Regression 0.699 ± 0.083 | GRNN 0.699 ± 0.064 | RUSBoost 0.698 ± 0.092 | TabNet 0.671 ± 0.138
Καλύτερο ROC-AUC: SVM 0.854 ± 0.069. Χειρότερο και πιο ασταθές: TabNet 0.707 ± 0.196.

WDBC (569 δείγματα): Accuracy μέσος ± απόκλιση
MLP 0.977 ± 0.010 | Logistic Regression 0.974 ± 0.019 | SVM 0.965 ± 0.021 | kNN 0.963 ± 0.020 | RUSBoost 0.960 ± 0.016 | Autoencoder 0.960 ± 0.022 | TabNet 0.958 ± 0.013 | GRNN 0.939 ± 0.026

Συμπεράσματα:
- Coimbra: οι αποκλίσεις (5-14%) είναι ίδιου μεγέθους με τις διαφορές των μοντέλων, άρα δεν ξεχωρίζει σαφώς κανένα.
- Το πλεονέκτημα του TabNet στον ένα χωρισμό δεν επιβεβαιώθηκε (βγήκε τελευταίο, ασταθές).
- WDBC: όλα 94-98%. Τα deep learning δεν ξεπερνούν σαφώς τη Logistic Regression.
- Η βελτίωση Coimbra → WDBC παρατηρείται και στα κλασικά μοντέλα, άρα οφείλεται στο dataset και όχι στα deep learning.

Περιορισμοί: μία επανάληψη 5-fold. Σταθερές ρυθμίσεις TabNet που επέλεξα βλέποντας τον πρώτο χωρισμό. Στο Coimbra το early stopping γίνεται σε validation set 18-19 δειγμάτων. Κλασικά μοντέλα χωρίς tuning. Το XAI δεν έγινε ανά fold.

## 6/10 - XAI και στα 8 μοντέλα: SHAP, permutation, LIME (xai_all_models.py)

Ίδιες τρεις τεχνικές και στα 8 μοντέλα (4 ML + 4 DL), σε Coimbra και WDBC, στα μοντέλα του πρώτου χωρισμού.
LIME: μέσος όρος απόλυτων συντελεστών, 24 δείγματα test (Coimbra) και 40 από 114 (WDBC), 1000 δείγματα ανά εξήγηση, random_state=42.
Αποτελέσματα: results_xai_all/ (shap_, permutation_, lime_, agreement_ CSV και bar plots SHAP).

Συμφωνία τεχνικών (Spearman ρ):
- Coimbra: SHAP-LIME 0.68-0.93, SHAP-Permutation 0.48-0.82, Permutation-LIME 0.10-0.78.
- WDBC: SHAP-LIME 0.89-0.95. Το permutation διαφωνεί στα kNN (ρ 0.01), GRNN (0.16), RUSBoost (0.30).

Ευρήματα:
- Coimbra: Glucose στο top-5 σε 23/24 περιπτώσεις (8 μοντέλα x 3 τεχνικές), BMI σε 22/24. Resistin στο top-5 και των τριών τεχνικών σε όλα εκτός από τον Autoencoder (κυριαρχούν HOMA και Insulin).
- Age: στο top-5 του permutation σε 7/8 μοντέλα, του SHAP σε 6/8, του LIME μόνο στο kNN. Δεν μπορώ να πω ποια τεχνική έχει δίκιο (24 δείγματα test).
- RUSBoost (Coimbra): SHAP μη μηδενικό μόνο σε 3 χαρακτηριστικά, το 4ο-5ο της λίστας είναι ισοβαθμίες.
- WDBC: τουλάχιστον ένα χαρακτηριστικό μεγέθους (radius/perimeter/area) στο top-5 του SHAP και των 8 μοντέλων. worst texture σε 7/8, worst concave points σε 6/8.

Σημείωση: το SVM δίνει Accuracy 0.750 με όριο 0.5 στο predict_proba και 0.708 με predict() (train_models.py), ίδιο ROC-AUC 0.797.

Περιορισμοί: μία εκτέλεση, χωρίς διακύμανση. Το Spearman σε 9 χαρακτηριστικά (Coimbra) είναι θορυβώδες. Δεν έχω ελέγξει τις τιμές του permutation για kNN και RUSBoost στο WDBC.