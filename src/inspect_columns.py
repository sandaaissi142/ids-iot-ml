import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)

df = pd.read_csv("data/train_test_network.csv")

print("Valeurs manquantes (NaN) par colonne :")
nan = df.isna().sum()
print(nan[nan > 0])
print()

print("Colonnes texte : nombre de valeurs uniques et part de '-' :")
rows = []
for col in df.select_dtypes(exclude="number").columns:
    rows.append({
        "colonne": col,
        "uniques": df[col].nunique(),
        "part_tiret_%": round((df[col] == "-").mean() * 100, 1),
    })
print(pd.DataFrame(rows).sort_values("uniques").to_string(index=False))
print()

print("Colonnes numeriques constantes (inutiles) :")
num = df.select_dtypes(include="number")
print([c for c in num.columns if num[c].nunique() <= 1])
