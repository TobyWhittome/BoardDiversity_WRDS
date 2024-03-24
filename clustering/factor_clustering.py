import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Factor Loadings from previous factor analysis
factor_loadings = np.array([
    [-0.0268391,  0.04875818,  0.29395359, -0.02225198],
    [-0.03766627, -0.03551647, -0.14508402,  0.03103545],
    [-0.02905915, -0.26168191,  0.93544646,  0.10232526],
    [-0.01348789,  0.27345831, -0.29850815,  0.91306802],
    [ 0.78948211,  0.00878194,  0.02262549, -0.00574108],
    [ 0.94775419, -0.0115682,   0.05393426, -0.02288017],
    [ 0.00625683,  0.99694649, -0.03270356, -0.01272129],
    [-0.02257146,  0.10773019, -0.21646061,  0.02004274],
    [ 0.00647368,  0.07445822, -0.03491391, -0.30278987]
])

# Perform K-means clustering on factor loadings
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(factor_loadings)

# Elbow method to find the optimal number of clusters, with adjusted range
sse = []
max_clusters = len(factor_loadings) - 1  # Maximum number of clusters is one less than number of samples
for k in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(factor_loadings)
    sse.append(kmeans.inertia_)

# Plot SSE for each *k*
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_clusters + 1), sse, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Calculate the silhouette score for the initially chosen number of clusters
silhouette_avg = silhouette_score(factor_loadings, clusters)
print(f'The average silhouette score for the clusters is: {silhouette_avg}')
