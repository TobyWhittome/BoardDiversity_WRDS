from itertools import combinations
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load your dataset
df = pd.read_excel('final_dataset.xlsx')

# Original list of columns for clustering
original_columns = ['high_voting_power', 'percentage_INEDs', 'num_directors_>4.5', 'total_share_%', 'total_memberships', 'boardsize', 'CEODuality', 'dualclass']

# Dictionary to store silhouette scores and corresponding columns
silhouette_scores = {}

# Generate all combinations of columns to exclude 2 at a time
for combo in combinations(original_columns, 2):
    columns_to_exclude = list(combo)
    columns_to_cluster = [col for col in original_columns if col not in columns_to_exclude]  # Exclude the current combo

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[columns_to_cluster])

    # Perform K-means clustering with an appropriate number of clusters
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)

    # Calculate silhouette score
    score = silhouette_score(data_scaled, clusters)
    silhouette_scores[', '.join(columns_to_cluster)] = score
    
for columns, score in silhouette_scores.items():
    print(f"Columns: {columns} --> Silhouette Score: {score}")

# Identify the highest silhouette score and corresponding columns
best_columns = max(silhouette_scores, key=silhouette_scores.get)
best_score = silhouette_scores[best_columns]

print("\nBest Configuration:")
print(f"Columns: {best_columns}")
print(f"Silhouette Score: {best_score}")
