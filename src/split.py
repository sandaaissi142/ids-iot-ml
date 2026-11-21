import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed.csv")
print("Lignes avant :", len(df))

avant = len(df)
df = df.drop_duplicates()
print("Doublons retires :", avant - len(df))
print("Lignes apres :", len(df))

train, test = train_test_split(
    df, test_size=0.2, stratify=df["type"], random_state=42
)

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)

print()
print("Train :", train.shape, " Test :", test.shape)
print()
repartition = pd.concat(
    [train["type"].value_counts().rename("train"),
     test["type"].value_counts().rename("test")],
    axis=1,
)
print(repartition)