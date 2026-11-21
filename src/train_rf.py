import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

cibles = ["label", "type"]
X_train, y_train = train.drop(columns=cibles), train["type"]
X_test, y_test = test.drop(columns=cibles), test["type"]

modele = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42
)
print("Entrainement en cours...")
modele.fit(X_train, y_train)

pred = modele.predict(X_test)

acc = accuracy_score(y_test, pred)
f1_macro = f1_score(y_test, pred, average="macro")
rapport = classification_report(y_test, pred, digits=3)

print()
print(f"Accuracy : {acc:.4f}")
print(f"F1 macro : {f1_macro:.4f}")
print()
print(rapport)

with open("results/rf_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy : {acc:.4f}\nF1 macro : {f1_macro:.4f}\n\n{rapport}")

classes = sorted(y_test.unique())
cm = confusion_matrix(y_test, pred, labels=classes)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Prediction")
plt.ylabel("Reel")
plt.title("Matrice de confusion - Random Forest")
plt.tight_layout()
plt.savefig("results/confusion_rf.png", dpi=150)

joblib.dump(modele, "models/random_forest.joblib")
print("Fichiers : results/rf_report.txt, results/confusion_rf.png")