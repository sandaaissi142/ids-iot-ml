import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

test = pd.read_csv("data/test.csv")
X_test = test.drop(columns=["label", "type"])
y_test = test["type"]

rf = joblib.load("models/random_forest.joblib")

# 1. Variables les plus importantes
imp = pd.Series(rf.feature_importances_, index=X_test.columns)
imp = imp.sort_values(ascending=False)
print("Top 15 des variables les plus importantes :")
print(imp.head(15).round(4))

ax = imp.head(15).sort_values().plot(kind="barh", figsize=(8, 6), color="#2a6fdb")
ax.set_title("Top 15 des variables - Random Forest")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("results/importance_rf.png", dpi=150)

# 2. Erreurs les plus frequentes
pred = rf.predict(X_test)
res = pd.DataFrame({"reel": y_test.values, "predit": pred})
err = res[res["reel"] != res["predit"]]

print()
print("Erreurs les plus frequentes (reel -> predit) :")
print(err.value_counts().head(10))

# 3. Focus sur mitm
mitm = err[(err["reel"] == "mitm") | (err["predit"] == "mitm")]
print()
print("Confusions impliquant mitm :")
print(mitm.value_counts())

print()
print("Graphique : results/importance_rf.png")