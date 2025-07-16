import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────────
def load_data(csv_path="trojan_features.csv"):
    df = pd.read_csv(csv_path)
    y = df['label'].values
    X = df.drop(columns=['file', 'signal', 'label', 'auto_label'], errors='ignore')
    return X, y

# ──────────────────────────────────────────────────────────────
# 2. EVALUATE + FINAL TRAIN (OPTIMIZED)
# ──────────────────────────────────────────────────────────────
def evaluate_cv(X, y, n_splits=3, n_repeats=3):
    start = time.time()

    clf = RandomForestClassifier(
        n_estimators=80,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    proba_sum = np.zeros_like(y, dtype=float)

    for seed in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        probas = cross_val_predict(clf, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
        proba_sum += probas

    y_proba = proba_sum / n_repeats
    y_pred  = (y_proba >= 0.5).astype(int)

    print("✅ ROC AUC:", round(roc_auc_score(y, y_proba), 3))
    print("\n🧾 Classification Report:\n", classification_report(y, y_pred, target_names=["Trusted", "Trojan"]))

    cm = confusion_matrix(y, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Trusted", "Trojan"]).plot(cmap=plt.cm.Blues)
    plt.tight_layout()
    # plt.show()

    clf.fit(X, y)
    print(f"✅ Finished training in {round(time.time() - start, 2)} seconds")
    return clf

# ──────────────────────────────────────────────────────────────
# 3. GINI FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────────
def plot_gini_importance(clf, X, top_k=8):
    if not hasattr(clf, "feature_importances_"):
        print("❌ Classifier not trained.")
        return

    gini_imp = pd.Series(clf.feature_importances_, index=X.columns)
    top_features = gini_imp.sort_values(ascending=False).head(top_k)

    plt.figure(figsize=(6, 4))
    top_features.plot.barh(color='skyblue')
    plt.title("Top Gini Feature Importances")
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.tight_layout()
    # plt.show()

# ──────────────────────────────────────────────────────────────
# 4. SAVE/LOAD MODEL
# ──────────────────────────────────────────────────────────────
def save_pipeline(clf, scaler=None, folder="model"):
    os.makedirs(folder, exist_ok=True)
    joblib.dump(clf, os.path.join(folder, "rf_trojan_detector.joblib"))
    if scaler:
        joblib.dump(scaler, os.path.join(folder, "scaler.joblib"))
    print(f"✅ Model & scaler saved to ./{folder}/")

def load_pipeline(folder="model"):
    clf = joblib.load(os.path.join(folder, "rf_trojan_detector.joblib"))
    scaler = None
    scaler_path = os.path.join(folder, "scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    return clf, scaler

# ──────────────────────────────────────────────────────────────
# 5. PREDICT NEW DUMP
# ──────────────────────────────────────────────────────────────
def predict_new_dump(clf, scaler, feats: dict):
    X_new = pd.DataFrame([feats])
    if scaler:
        X_new[X_new.columns] = scaler.transform(X_new[X_new.columns])
    prob = clf.predict_proba(X_new)[:, 1][0]
    pred = int(prob > 0.5)
    print(f"→ Trojan-prob = {prob:.2f}, label = {pred}")
    return prob, pred

# ──────────────────────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Load data
    X, y = load_data("trojan_features.csv")
    print("📊 Feature shape:", X.shape)

    # 2) Standardize
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 3) Train & evaluate
    clf = evaluate_cv(X_scaled, y, n_splits=3, n_repeats=3)

    # 4) Show top features
    plot_gini_importance(clf, X_scaled)

    # 5) Save model
    save_pipeline(clf, scaler)
    clf, scaler = load_pipeline("model")

    # 6) Future inference
    # feats = extract_features_from_vcd("vcd_dumps/new.vcd")  # your custom extractor
    # predict_new_dump(clf, scaler, feats)
