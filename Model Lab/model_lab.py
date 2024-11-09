import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=50, c='gray', edgecolor='k', alpha=0.6)
plt.title("Generated Data Points (Before Clustering)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
y_kmeans = kmeans.fit_predict(X)

centers = kmeans.cluster_centers_

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis', edgecolor='k', alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', edgecolor='k', label="Cluster Centers")
plt.colorbar(scatter, label="Cluster Label")
plt.title("K-Means Clustering with Cluster Centers")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

for i, center in enumerate(centers):
    plt.text(center[0], center[1], f'Center {i+1}', color='red', fontsize=12, ha='center', va='center', weight='bold')

plt.legend()
plt.grid(True)
plt.show()

print("Cluster centers:\n", centers)

