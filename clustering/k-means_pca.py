import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_excel('dataset/transformed_dataset.xlsx')
#columns_to_cluster = ['genderratio', 'nationalitymix', 'voting_power', 'percentage_INEDs', 'num_directors_>4.5', 'total_share_%', 'boardsize', 'CEODuality', 'dualclass']

columns_to_cluster = ['voting_power', 'percentage_INEDs', 'num_directors_>4.5', 'total_share_%', 'CEODuality', 'dualclass', 'boardsize_mean']

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df[columns_to_cluster])
print(data_scaled.shape)

# Apply PCA

#0.87 gets rid of 2 dimensions and has 0.319 for n=4
pca = PCA(n_components=0.87)  # Retain 87% of the variance
data_pca = pca.fit_transform(data_scaled)
print(data_pca.shape)

# Perform K-means clustering
# Adjust the number of clusters (n_clusters) based on your dataset
data_pca = data_scaled
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

# Convert SSE data into a DataFrame for Seaborn plotting
sse_df = pd.DataFrame({'Number of clusters': range(1, 11), 'SSE': sse})
print(sse_df)
sse_df.replace([np.inf, -np.inf], np.nan, inplace=True)


# Calculate the silhouette score
silhouette_avg = silhouette_score(data_pca, clusters)
print(f'The average silhouette score for the clusters is: {silhouette_avg}')
