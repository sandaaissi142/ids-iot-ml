import os
import time
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
cibles = ["label", "type"]
X_train, y_train = train.drop(columns=cibles), train["type"]
X_test, y_test = test.drop(columns=cibles), test["type"]

# Modele complet (tache 12), un seul coeur pour une comparaison equitable
rf_full = joblib.load("models/random_forest.joblib")
rf_full.set_params(n_jobs=1)

# 15 variables les plus importantes
imp = pd.Series(rf_full.feature_importances_, index=X_train.columns)
top15 = list(imp.sort_values(ascending=False).head(15).index)
print("15 variables retenues :")
print(top15)

rf_light = RandomForestClassifier(
    n_estimators=30, max_depth=12, class_weight="balanced",
    n_jobs=1, random_state=42,
).fit(X_train[top15], y_train)

arbre = DecisionTreeClassifier(
    max_depth=10, class_weight="balanced", random_state=42,
).fit(X_train[top15], y_train)


def mesurer(nom, modele, cols, chemin):
    joblib.dump(modele, chemin)
    taille_ko = os.path.getsize(chemin) / 1024
    debut = time.perf_counter()
    pred = modele.predict(X_test[cols])
    duree = time.perf_counter() - debut
    ms_pour_1000 = duree / len(X_test) * 1000 * 1000
    return {
        "modele": nom,
        "variables": len(cols),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "f1_macro": round(f1_score(y_test, pred, average="macro"), 4),
        "f1_mitm": round(f1_score(y_test, pred, labels=["mitm"], average=None)[0], 4),
        "taille_Ko": round(taille_ko),
        "ms_pour_1000": round(ms_pour_1000, 1),
    }


tous = list(X_train.columns)
lignes = [
    mesurer("RF complet", rf_full, tous, "models/rf_full_copy.joblib"),
    mesurer("RF leger", rf_light, top15, "models/rf_light.joblib"),
    mesurer("Arbre de decision", arbre, top15, "models/tree.joblib"),
]
tableau = pd.DataFrame(lignes)
print()
print(tableau.to_string(index=False))
tableau.to_csv("results/comparaison_legers.csv", index=False)
print()
print("Fichier : results/comparaison_legers.csv")