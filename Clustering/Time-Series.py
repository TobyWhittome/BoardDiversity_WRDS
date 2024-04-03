import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import matplotlib.lines as mlines

# Function to process a single year's dataset
def process_dataset(df, columns_to_cluster):
    data_scaled = StandardScaler().fit_transform(df[columns_to_cluster])
    data_scaled = PCA(n_components=0.83).fit_transform(data_scaled)
    print(data_scaled.shape)
    return data_scaled
  
def load_and_truncate(filepath, min_length):
    df = pd.read_excel(filepath)
    # Truncate the dataset if it's longer than min_length
    if len(df) > min_length:
        df = df.iloc[:min_length]
    return df


years = ['2007', '2008', '2009', '2010', '2011', '2012', '2013','2014','2015', '2016', '2017', '2018', '2019','2020', '2021', '2022', '2023']
files = [f'prev_data/{year}_dataset.xlsx' for year in years]
columns_to_cluster = ['genderratio', 'nationalitymix', 'voting_power', 'percentage_INEDs', 'num_directors_>4.5', 'total_share_%', 'boardsize', 'CEODuality', 'dualclass']

# Find the number of rows in the smallest dataset
min_length = min(len(pd.read_excel(file)) for file in files)

# Load and truncate datasets
datasets = [load_and_truncate(file, min_length)[columns_to_cluster] for file in files]

# Process each dataset and collect the PCA-transformed data
datasets_scaled = np.array([process_dataset(df, columns_to_cluster) for df in datasets])

# Scale the data (important for DTW)
datasets_scaled = TimeSeriesScalerMeanVariance().fit_transform(datasets)

# Time-series clustering using DTW and KMeans
n_clusters = 3
model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", verbose=True, random_state=42)
clusters = model.fit_predict(datasets_scaled)


def get_year_ranges(years):
    years = sorted(set(int(year) for year in years))
    ranges = []
    start = years[0]

    for i in range(1, len(years)):
        if years[i] != years[i-1] + 1:
            end = years[i-1]
            if start == end:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{end}")
            start = years[i]
    ranges.append(f"{start}-{years[-1]}" if start != years[-1] else f"{start}")
    return ranges


plt.figure(figsize=(8, 8))
clusternames = ['A', 'B', 'C']
cluster_colours = ['#b4d2b1', '#568f8b', '#1d4a60']
cluster_colours = ['#f24834', '#fb8c6c', '#c0151a']


for yi in range(n_clusters):
    plt.subplot(n_clusters, 1, yi + 1)
    plotted_years = []
    
    cluster_series_indices = np.where(clusters == yi)[0]
    
    for ix in cluster_series_indices:
        plt.plot(datasets_scaled[ix].ravel(), "k-", alpha=0.2)
        plotted_years.append(years[ix])
    
    plt.plot(model.cluster_centers_[yi].ravel(), f'{cluster_colours[yi]}', label='Cluster Center')
    plt.xlim(0, datasets_scaled.shape[1])
    plt.ylim(-4, 4)
    plt.title(f"Cluster {clusternames[yi]}", fontweight='bold')
    
    year_ranges_label = f"Years: {', '.join(get_year_ranges(plotted_years))}"
    
    black_line = mlines.Line2D([], [], color='black', label=year_ranges_label)
    red_line = mlines.Line2D([], [], color=f'{cluster_colours[yi]}', label='Cluster Center')
    plt.legend(handles=[black_line, red_line], loc="upper left", prop={'weight':'bold'})

plt.subplots_adjust(left=0.03, right=0.98, top=0.95, bottom=0.05, hspace=0.5)
plt.show()