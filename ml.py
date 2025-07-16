import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# 1) Load your raw dataset
df = pd.read_csv("trojan_dataset_combined.csv")

# 2) Select features & label
#    Drop file/signal columns; keep numeric features only
drop_cols = ["file","signal","signal_type","label","auto_label"]
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols].values
y = df["label"].values

# 3) Train/test split with stratification on y
#    This ensures both classes appear in train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 4) Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 5) Train RandomForest with class_weight='balanced'
clf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)
clf.fit(X_train, y_train)

# 6) Evaluate
y_pred    = clf.predict(X_test)
y_proba   = clf.predict_proba(X_test)[:,1]  # probability of class 1
acc       = accuracy_score(y_test, y_pred)
auc       = roc_auc_score(y_test, y_proba)

print(f"\nAccuracy on test set: {acc:.3f}")
print(f"AUC on test set:      {auc:.3f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Trusted","Trojan"]))

# 7) Show predicted probabilities for each test sample
probs_df = pd.DataFrame({
    "true_label": y_test,
    "pred_label": y_pred,
    "prob_trojan": y_proba
})
print("\nSample of test‑set predicted probabilities:")
print(probs_df.head(10))

#from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)
import matplotlib.pyplot as plt
import numpy as np

# Suppose X (features) and y (labels) are your full dataset
clf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)

# 5‑fold stratified CV, but get out‑of‑fold predictions so we can compute a confusion matrix
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
y_pred = cross_val_predict(clf, X, y, cv=skf)
y_proba = cross_val_predict(clf, X, y, cv=skf, method='predict_proba')[:,1]

auc = roc_auc_score(y, y_proba)
print("Overall ROC AUC:", round(auc, 3))
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=['Trusted','Trojan']))

# Plot confusion matrix
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Trusted','Trojan'])
disp.plot(cmap=plt.cm.Blues)
plt.show()

