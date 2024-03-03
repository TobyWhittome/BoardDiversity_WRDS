import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Load the dataset
df = pd.read_excel('final_dataset.xlsx')

# Preprocess the data
# Assuming no missing values or handling them beforehand
# Select the numerical columns for clustering
columns_to_cluster = ['high_voting_power', 'percentage_INEDs', 'num_directors_>4.5', 'total_share_%', 'total_memberships', 'boardsize', 'CEODuality', 'dualclass']
#test_columns_to_cluster = ['percentage_INEDs', 'num_directors_>4.5', 'total_share_%', 'total_memberships', 'boardsize', 'dualclass']

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df[columns_to_cluster])

# Implement K-means
# Start with k=3 for demonstration; adjust after using the Elbow method if needed
#3 or 5 is good here
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Add the cluster labels to your original dataframe
df['Cluster'] = clusters

# Summary statistics for each cluster
for i in range(4):  # Adjust based on the number of clusters chosen
    print(f"Cluster {i+1} Summary Statistics")
    print(df[df['Cluster'] == i].describe())
    print("\n")

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

silhouette_avg = silhouette_score(data_scaled, df['Cluster'])
print(f'The average sil score for the clusters is {silhouette_avg}')