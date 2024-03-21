import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

#Useful if noise is in the data, and want to find clusters of any shape

# Load the dataset
df = pd.read_excel('final_dataset.xlsx')

# Preprocess the data
# Assuming no missing values or handling them beforehand
# Select the numerical columns for clustering, excluding performance variables
columns_to_cluster = ['high_voting_power', 'percentage_INEDs', 'num_directors_>4.5', 'total_share_%', 'total_memberships', 'boardsize', 'CEODuality', 'dualclass']

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df[columns_to_cluster])

# Use NearestNeighbors to estimate a good eps value
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(data_scaled)
distances, indices = neighbors_fit.kneighbors(data_scaled)

# Sort distance values by ascending value and plot
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.plot(distances)
plt.title('K-Nearest Neighbors Distances')
plt.xlabel('Points sorted by distance')
plt.ylabel('Epsilon distance')
#plt.show()

# You should see a noticeable "elbow" in the plot, that's a good value for eps.
# Assuming an elbow at eps_value based on the plot
#Best value for highest silhouette score is 1.4-1.5 here.
eps_value = 1.5 # <- Set this based on your k-distance graph

# Apply DBSCAN with the estimated eps and a min_samples value
dbscan = DBSCAN(eps=eps_value, min_samples=5) # Adjust min_samples based on domain knowledge or heuristic
db_clusters = dbscan.fit_predict(data_scaled)

# Add the cluster labels to your original dataframe
df['DBSCAN_Cluster'] = db_clusters

# Now you can analyze your clusters
print(df.groupby('DBSCAN_Cluster').size())

# If needed, you can proceed with further analysis of the clusters

silhouette_avg = silhouette_score(data_scaled, df['DBSCAN_Cluster'])
print(f'The average sil score for the clusters is {silhouette_avg}')
