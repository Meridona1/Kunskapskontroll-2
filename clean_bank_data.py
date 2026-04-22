import pandas as pd

# Läs in data
df = pd.read_csv("bank_data.csv")

# Visa info om datan
print("FÖRE TVÄTT:")
print(df.info())

# Ta bort dubbletter
df = df.drop_duplicates()

# Ta bort rader med saknade värden (NaN)
df = df.dropna()

# Visa info efter tvätt
print("\nEFTER TVÄTT:")
print(df.info())

# Spara ren data
df.to_csv("bank_data_clean.csv", index=False)

print("\nKlar! Ren data sparad som bank_data_clean.csv") 