import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Сгенерируем случайные данные для кластеризации
np.random.seed(0)
n_samples = 200
n_clusters = 3
X = np.random.randn(n_samples, 2)

# Применяем алгоритм K-средних для кластеризации данных
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Визуализация результатов кластеризации
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], marker='x', color='red', s=200)
plt.title('K-means Clustering')
plt.show()
