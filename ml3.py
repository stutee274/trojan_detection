from joblib import load
import pprint

clf = load("model/rf_trojan_detector.joblib")
print(clf)  # or pprint.pprint(vars(clf)) for full info

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from joblib import dump

df = pd.read_csv("trojan_dataset_combined.csv")  # your extracted features

X = df.drop(columns=["label", "file", "signal_type","signal"])  # features only
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=80, max_depth=5, 
        random_state=42, class_weight ='balanced',
        n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

dump(clf, "model/rf_trojan_detector.joblib")  # update model
