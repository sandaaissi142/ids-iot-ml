import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

df = pd.read_csv("data/train_test_network.csv")

print("Dimensions :", df.shape)
print()
print("Premières lignes (transposées) :")
print(df.head(3).T)
print()
print("Types des colonnes :")
df.info()