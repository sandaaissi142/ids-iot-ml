import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/train_test_network.csv")

print("Colonnes :")
print(list(df.columns))
print()

print("Normal (0) / Attaque (1) :")
print(df["label"].value_counts())
print()
print("En pourcentage :")
print((df["label"].value_counts(normalize=True) * 100).round(1))
print()

print("Types de trafic :")
print(df["type"].value_counts())

counts = df["type"].value_counts()
ax = counts.plot(kind="barh", figsize=(8, 5), color="#2a6fdb")
ax.invert_yaxis()
ax.set_title("Repartition des types de trafic - ToN-IoT")
ax.set_xlabel("Nombre d'echantillons")
plt.tight_layout()
plt.savefig("results/repartition_types.png", dpi=150)
print()
print("Graphique enregistre : results/repartition_types.png")