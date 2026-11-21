import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

cibles = ["label", "type"]
X_train, y_train = train.drop(columns=cibles), train["type"]
X_test, y_test = test.drop(columns=cibles), test["type"]

# XGBoost demande des classes numeriques
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

poids = compute_sample_weight("balanced", y_train_enc)
xgb = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                    tree_method="hist", n_jobs=-1, random_state=42)
print("Entrainement XGBoost...")
xgb.fit(X_train, y_train_enc, sample_weight=poids)
pred_xgb = le.inverse_transform(xgb.predict(X_test))

# Random Forest deja entraine (tache 12)
rf = joblib.load("models/random_forest.joblib")
pred_rf = rf.predict(X_test)

lignes = []
for nom, p in [("Random Forest", pred_rf), ("XGBoost", pred_xgb)]:
    lignes.append({
        "modele": nom,
        "accuracy": round(accuracy_score(y_test, p), 4),
        "f1_macro": round(f1_score(y_test, p, average="macro"), 4),
        "f1_mitm": round(f1_score(y_test, p, labels=["mitm"], average=None)[0], 4),
    })
tableau = pd.DataFrame(lignes)
print()
print(tableau.to_string(index=False))
tableau.to_csv("results/comparaison_modeles.csv", index=False)

rapport = classification_report(y_test, pred_xgb, digits=3)
print()
print(rapport)
with open("results/xgb_report.txt", "w", encoding="utf-8") as f:
    f.write(rapport)

classes = sorted(y_test.unique())
cm = confusion_matrix(y_test, pred_xgb, labels=classes)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Prediction")
plt.ylabel("Reel")
plt.title("Matrice de confusion - XGBoost")
plt.tight_layout()
plt.savefig("results/confusion_xgb.png", dpi=150)

joblib.dump(xgb, "models/xgboost.joblib")
joblib.dump(le, "models/label_encoder.joblib")
print("Fichiers : comparaison_modeles.csv, xgb_report.txt, confusion_xgb.png")