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