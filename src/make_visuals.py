import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results/comparaison_legers.csv")

noms = ["RF complet\n(52 variables)",
        "RF l\u00e9ger\n(15 variables)",
        "Arbre de d\u00e9cision\n(15 variables)"]
couleurs = ["#1F4E9C", "#2a6fdb", "#7fb0f0"]


def fr(x, n=1):
    return f"{x:.{n}f}".replace(".", ",")


def taille(ko):
    return fr(ko / 1024, 1) + " Mo" if ko >= 1024 else f"{int(ko)} Ko"


acc = df["accuracy"] * 100
ko = df["taille_Ko"]
ms = df["ms_pour_1000"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
y = [0, 1, 2]

# 1. Precision
ax = axes[0]
ax.barh(y, acc, color=couleurs)
for i, v in enumerate(acc):
    ax.text(v + 1, i, fr(v, 2) + " %", va="center", fontsize=10)
ax.set_xlim(0, 120)
ax.set_title("Pr\u00e9cision (accuracy)", fontsize=12, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels(noms, fontsize=10)
ax.invert_yaxis()

# 2. Taille
ax = axes[1]
ax.barh(y, ko, color=couleurs)
ax.set_xscale("log")
ax.set_xlim(10, 2e7)
for i, v in enumerate(ko):
    ax.text(v * 1.3, i, taille(v), va="center", fontsize=10)
ax.set_title("Taille du mod\u00e8le (\u00e9chelle log)", fontsize=12, fontweight="bold")

# 3. Vitesse
ax = axes[2]
ax.barh(y, ms, color=couleurs)
ax.set_xscale("log")
ax.set_xlim(0.1, 1000)
for i, v in enumerate(ms):
    ax.text(v * 1.3, i, fr(v, 1) + " ms", va="center", fontsize=10)
ax.set_title("Temps pour 1000 pr\u00e9dictions (log)", fontsize=12, fontweight="bold")

for a in axes:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

fig.suptitle("D\u00e9tecteur d'intrusions IoT : pr\u00e9cision, taille et vitesse",
             fontsize=15, fontweight="bold")
fig.text(0.5, 0.01,
         "Dataset ToN-IoT (UNSW Canberra) - jeu de test : 21 067 \u00e9chantillons",
         ha="center", fontsize=9, color="#555555")
plt.tight_layout(rect=[0, 0.04, 1, 0.94])
plt.savefig("results/comparaison_visuelle.png", dpi=200)
print("Graphique : results/comparaison_visuelle.png")