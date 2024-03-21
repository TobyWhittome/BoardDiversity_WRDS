import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load the dataset
df = pd.read_excel('final_dataset.xlsx')

# Original list of columns for clustering
original_columns = ['high_voting_power', 'percentage_INEDs', 'num_directors_>4.5', 'total_share_%', 'total_memberships', 'boardsize', 'CEODuality', 'dualclass']

# Dictionary to store silhouette scores and corresponding columns
silhouette_scores = {}

# Iterate over the columns, excluding one column at a time
for column in original_columns:
    columns_to_cluster = [col for col in original_columns if col != column]  # Exclude the current column

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[columns_to_cluster])

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)  # Adjust n_clusters as needed
    clusters = kmeans.fit_predict(data_scaled)

    # Calculate silhouette score
    score = silhouette_score(data_scaled, clusters)
    silhouette_scores[', '.join(columns_to_cluster)] = score

# Print the silhouette scores and corresponding columns
for columns, score in silhouette_scores.items():
    print(f"Columns: {columns} --> Silhouette Score: {score}")

# Identify the highest silhouette score and corresponding columns
best_columns = max(silhouette_scores, key=silhouette_scores.get)
best_score = silhouette_scores[best_columns]

print("\nBest Configuration:")
print(f"Columns: {best_columns}")
print(f"Silhouette Score: {best_score}")
