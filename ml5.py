# ml_pipeline_improved.py
import os, time, glob, joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from features_extraction import process_vcd_files
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────
# 1. BUILD DATASET (ON‐THE‐FLY)
# ──────────────────────────────
def build_dataset(vcd_dir="vcd_dumps"):
    # Parse ALL VCDs into a DataFrame
    df = process_vcd_files(vcd_dir)
    # Save for inspection
    df.to_csv("trojan_dataset_combined.csv", index=False)
    print("✅ Saved combined dataset with", len(df), "rows → trojan_dataset_combined.csv")
    return df

# ──────────────────────────────
# 2. SELECT FEATURES
# ──────────────────────────────
def select_top_features(df, top_features=None):
    # Default to the top‐6 you identified visually
    if top_features is None:
        top_features = [
            "toggle_rate",
            "bitflip_rate",
            "toggles",
            "jump_max",
            "avg_jump",
            "glitches", "entropy"
        ]
    X = df[top_features]
    y = df["label"].values
    return X, y

# ──────────────────────────────
# 3. TRAIN + EVALUATE WITH K‐FOLD
# ──────────────────────────────
def train_and_evaluate(X, y, n_splits=3, n_repeats=3):
    start = time.time()
    clf = RandomForestClassifier(
        n_estimators=80,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    proba_sum = np.zeros_like(y, dtype=float)

    for seed in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        probas = cross_val_predict(
            clf, X, y, cv=cv,
            method="predict_proba", n_jobs=-1
        )[:, 1]
        proba_sum += probas

    y_proba = proba_sum / n_repeats
    y_pred  = (y_proba >= 0.5).astype(int)

    # Metrics
    print("✅ ROC AUC:", round(roc_auc_score(y, y_proba), 3))
    print("\n🧾 Classification Report:\n",
          classification_report(y, y_pred, target_names=["Trusted", "Trojan"]))

    cm = confusion_matrix(y, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Trusted","Trojan"])\
      .plot(cmap=plt.cm.Blues)
    plt.tight_layout()
    # plt.show()

    # K‑Fold accuracy scores
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=kf, n_jobs=-1)
    print("📊 CV accuracies:", np.round(scores,3), "→ mean:", round(scores.mean(),3))

    # Final fit on all data
    clf.fit(X, y)
    print(f"✅ Model trained in {round(time.time() - start,2)}s")
    return clf

# ──────────────────────────────
# 4. FEATURE IMPORTANCE PLOT
# ──────────────────────────────
def plot_importances(clf, feature_names):
    imp = pd.Series(clf.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=True)
    plt.figure(figsize=(6,4))
    imp.plot.barh(color="steelblue")
    plt.title("Final Gini Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    # plt.show()

# ──────────────────────────────
# 5. SAVE / LOAD PIPELINE
# ──────────────────────────────
def save_pipeline(clf, scaler, folder="model"):
    os.makedirs(folder, exist_ok=True)
    joblib.dump(clf,   os.path.join(folder, "rf_trojan.pkl"))
    joblib.dump(scaler,os.path.join(folder, "scaler.pkl"))
    print(f"✅ Pipeline saved to '{folder}/'")

def load_pipeline(folder="model"):
    clf    = joblib.load(os.path.join(folder, "rf_trojan.pkl"))
    scaler = joblib.load(os.path.join(folder, "scaler.pkl"))
    return clf, scaler

# ──────────────────────────────
# 6. INFERENCE ON NEW DUMPS
# ──────────────────────────────
def infer_on_new(vcd_dir="vcd_dumps"):
    clf, scaler = load_pipeline()
    vt = process_vcd_files(vcd_dir)  # regenerate features for all
    X_new, _ = select_top_features(vt)
    X_new = pd.DataFrame(scaler.transform(X_new), columns=X_new.columns)
    probs = clf.predict_proba(X_new)[:,1]
    preds = (probs>=0.5).astype(int)

    for fn, p, pr in zip(vt["file"], preds, probs):
        label = "TROJAN" if p else "TRUSTED"
        print(f"{fn:20s} → {label:7s} (p={pr:.2f})")

# ──────────────────────────────
# 7. MAIN
# ──────────────────────────────
if __name__ == "__main__":
    # 1) Build dataset from your VCDs
    df = build_dataset("vcd_dumps")

    # 2) Select only the 6 most‐important features
    X, y = select_top_features(df)

    # 3) Scale
    scaler = StandardScaler().fit(X)
    Xs = pd.DataFrame(scaler.transform(X), columns=X.columns)

    # 4) Train + evaluate
    clf = train_and_evaluate(Xs, y)

    # 5) Plot final importances
    plot_importances(clf, X.columns)

    # 6) Save pipeline
    save_pipeline(clf, scaler)

    # 7) Demonstrate inference on any new .vcd in vcd_dumps/
    infer_on_new("vcd_dumps")
