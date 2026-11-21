import pandas as pd

df = pd.read_csv("data/train_test_network.csv")

# Cibles
y_bin = df["label"]
y_multi = df["type"]

# Colonnes a retirer
cibles = ["label", "type"]
identifiants = ["src_ip", "dst_ip", "src_port"]
colonnes_texte = df.select_dtypes(exclude="number").columns.drop("type")
presque_vides = [c for c in colonnes_texte if (df[c] == "-").mean() >= 0.99]
trop_dispersees = ["dns_query"]

a_retirer = cibles + identifiants + presque_vides + trop_dispersees
X = df.drop(columns=a_retirer, errors="ignore")

# Encodage one-hot des colonnes texte restantes
X = pd.get_dummies(X, dtype=int)

print("Colonnes retirees (", len(a_retirer), ") :")
print(a_retirer)
print()
print("Caracteristiques finales :", X.shape)
print("Valeurs manquantes :", int(X.isna().sum().sum()))
print()
print(X.dtypes.value_counts())

# Sauvegarde (le dossier data/ n'est pas envoye sur GitHub)
sortie = X.assign(label=y_bin, type=y_multi)
sortie.to_csv("data/processed.csv", index=False)
print()
print("Fichier enregistre : data/processed.csv")