import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("bank_data_clean.csv")

X = df[["BALANCE", "PURCHASES"]].values

# Skala datan
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Skapa och träna KMeans
kmeans = KMeans(n_clusters=5, random_state=7)

y_pred = kmeans.fit_predict(X_scaled)

# Lägg till kluster som kolumn i dataframe
df["Cluster"] = y_pred

print(df.head())

df.groupby("Cluster").mean(numeric_only=True)


features = ["BALANCE", "PURCHASES"]
df.groupby("Cluster")[features].mean()


kmeans = KMeans(n_clusters=5, random_state=7)
y_pred = kmeans.fit_predict(X_scaled)

df["Cluster"] = y_pred

features = ["BALANCE", "PURCHASES"]
cluster_summary = df.groupby("Cluster")[features].mean()

print(cluster_summary)

print(df["Cluster"].value_counts())

plt.figure(figsize=(8, 6))

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df["Cluster"], s=10)

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=500)

plt.title("KMeans Clustering of Bank Customers")
plt.xlabel("Feature 1: Balance (scaled)")
plt.ylabel("Feature 2: Purchases (scaled)")

plt.show()

df.to_csv("bank_data_with_clusters.csv", index=False)
print("CSV skapad!")



