import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel('final_dataset.xlsx')

# Assuming all columns except the first (e.g., an ID or ticker column) are features

columns_to_cluster = ['high_voting_power', 'percentage_INEDs', 'num_directors_>4.5', 'total_share_%', 'total_memberships', 'boardsize', 'CEODuality', 'dualclass']

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df[columns_to_cluster])
print(data_scaled.shape)

# Apply PCA
#0.95 gets rid of one dimension and has 0.39 for n=5
#where as 0.9 gets rid of two dimensions and has 0.4 for n=5. Suggesting its better
#Three dimensions 0.43 -- 86% -- its 0.448 for n=4
#Four dimensions is 0.4646 -- 76% -- its 0.627 for n=4
#Five dimensions is 0.42
pca = PCA(n_components=0.86)  # Retain 95% of the variance
data_pca = pca.fit_transform(data_scaled)
print(data_pca.shape)

# Perform K-means clustering
# Adjust the number of clusters (n_clusters) based on your dataset
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(data_pca)

# Elbow method to find the optimal number of clusters
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(data_scaled)
    sse.append(kmeans.inertia_)

# Plot SSE for each *k*
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Calculate the silhouette score
silhouette_avg = silhouette_score(data_pca, clusters)
print(f'The average silhouette score for the clusters is: {silhouette_avg}')
