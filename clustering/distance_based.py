import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# Function to process a single year's dataset
def process_dataset(df, columns_to_cluster):
    data_scaled = StandardScaler().fit_transform(df[columns_to_cluster])
    PCAbosh = PCA(n_components=0.8).fit_transform(data_scaled)
    print(PCAbosh.shape)
    return PCAbosh
  
def load_and_truncate(filepath, min_length):
    df = pd.read_excel(filepath)
    # Truncate the dataset if it's longer than min_length
    if len(df) > min_length:
        df = df.iloc[:min_length]
    return df


years = ['2020', '2021', '2022']  # Add more years as needed
files = [f'prev_data/{year}_dataset.xlsx' for year in years]

columns_to_cluster = ['genderratio', 'nationalitymix', 'voting_power', 'percentage_INEDs', 'num_directors_>4.5', 'total_share_%', 'boardsize', 'CEODuality', 'dualclass']


# Find the number of rows in the smallest dataset
min_length = min(len(pd.read_excel(file)) for file in files)

# Load and truncate datasets
datasets = [load_and_truncate(file, min_length) for file in files]

# Process each dataset and collect the PCA-transformed data
datasets_pca = np.array([process_dataset(df, columns_to_cluster) for df in datasets])


# Scale the data (important for DTW)
datasets_scaled = TimeSeriesScalerMeanVariance().fit_transform(datasets_pca)

# Time-series clustering using DTW and KMeans
n_clusters = 4  # Adjust based on your data
model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", verbose=True, random_state=42)
clusters = model.fit_predict(datasets_scaled)

# Plotting the clusters
plt.figure(figsize=(12, 8))
for yi in range(n_clusters):
    plt.subplot(n_clusters, 1, 1 + yi)
    for xx in datasets_scaled[clusters == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(model.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, datasets_scaled.shape[1])
    plt.ylim(-4, 4)
    plt.title(f"Cluster {yi + 1}")

plt.tight_layout()
plt.show()


from mpl_toolkits.mplot3d import Axes3D

# Assuming your datasets_pca have 3 components for the 3D plot
# Flatten the datasets for plotting
flattened_data = np.concatenate(datasets_pca, axis=0)

# We also need to flatten the clusters list to match the size of flattened_data
flattened_clusters = np.concatenate([np.full(shape=dataset.shape[0], fill_value=cluster) for dataset, cluster in zip(datasets_pca, clusters)])

# Create a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Color map for the clusters
colors = ['r', 'g', 'b', 'y']

for cluster in range(n_clusters):
    # Get the data points that belong to the current cluster
    cluster_data = flattened_data[flattened_clusters == cluster]
    
    # Plot them with a specific color
    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], c=colors[cluster], label=f'Cluster {cluster + 1}')

# Labeling the axes
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')

# Adding a legend
ax.legend()

# Show plot
plt.show()
