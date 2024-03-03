import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
 
# Load the dataset
df = pd.read_excel('final_dataset.xlsx')
 
# Select the numerical columns for clustering (excluding 'tobinsQ' and 'mktcapitalisation')
columns_to_cluster = ['high_voting_power', 'percentage_INEDs', 'num_directors_>4.5', 'total_share_%', 'total_memberships', 'boardsize', 'CEODuality', 'dualclass']
 
# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df[columns_to_cluster])
 
# Choose the number of components for GMM
n_components = 3  # Replace with the number of clusters you wish to have
 
# Create a Gaussian Mixture Model
gmm = GaussianMixture(n_components=n_components, random_state=42)
 
# Fit the model and predict the clusters
gmm_clusters = gmm.fit_predict(data_scaled)
 
# Add the GMM cluster labels to your original dataframe
df['GMM_Cluster'] = gmm_clusters
 
# Calculate the silhouette score
silhouette_avg = silhouette_score(data_scaled, gmm_clusters)
print(f'The average silhouette score for the GMM clusters is: {silhouette_avg}')
 

