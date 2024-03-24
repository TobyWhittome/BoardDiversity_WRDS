from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from ds_utils.unsupervised import plot_cluster_cardinality, plot_cluster_magnitude, plot_magnitude_vs_cardinality
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

def get_silhouette_score(data):
  silhouette_scores = []
  for k in range(2, 7):
      km = KMeans(n_clusters=k, 
                  max_iter=300, 
                  tol=1e-04, 
                  init='k-means++', 
                  n_init=10, 
                  random_state=42)
      km.fit(data)
      silhouette_scores.append(silhouette_score(data, km.labels_))

  fig, ax = plt.subplots()
  ax.plot(range(2, 7), silhouette_scores, 'bx-')
  ax.set_title('Silhouette Score Method')
  ax.set_xlabel('Number of clusters')
  ax.set_ylabel('Silhouette Scores')
  plt.xticks(range(2, 7))
  plt.tight_layout()
  plt.show()


def magandcardinailty(data, km_fit):
  cluster_colors = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252']
  fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(12,4))

  #plot_cluster_cardinality(km_fit.labels_, ax=ax1, title="Cardinality", color=cluster_colors)
  plot_cluster_magnitude(data,
                        km_fit.labels_,
                        km_fit.cluster_centers_,
                        euclidean,
                        ax=ax2,
                        title="Magnitude",
                        color=cluster_colors
                        )
  plot_magnitude_vs_cardinality(data,
                                km_fit.labels_,
                                km_fit.cluster_centers_,
                                euclidean,
                                color=cluster_colors[0:km_fit.n_clusters],
                                ax=ax3, 
                                title="Magnitude vs. Cardinality")

  fig.autofmt_xdate(rotation=0)
  plt.tight_layout()
  plt.show()
  
  
def get_finaldset():
  df = pd.read_excel('dataset/transformed_dataset.xlsx')
  columns_to_cluster = ['voting_power', 'percentage_INEDs', 'num_directors_>4.5', 'total_share_%', 'CEODuality', 'dualclass', 'boardsize_mean']
  # Standardize the features
  scaler = StandardScaler()
  data_scaled = scaler.fit_transform(df[columns_to_cluster])
  print(data_scaled.shape)
  data_scaled = pd.DataFrame(data_scaled, columns=columns_to_cluster)
  return df[columns_to_cluster], data_scaled
  
def get_factor_loadings():
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
  return factor_loadings

def apply_PCA(data):
  #0.87 gets rid of 2 dimensions and has 0.319 for n=4
  pca = PCA(n_components=0.87)  # Retain 87% of the variance
  data_pca = pca.fit_transform(data)
  return data_pca

def ElbowVis(data):
  fig, ax = plt.subplots()
  visualizer = KElbowVisualizer(KMeans(), k=(2,7),ax=ax)
  visualizer.fit(data)

  ax.set_xticks(range(2,7))
  visualizer.show()
  plt.show()
  
""" def create_boxplot():
  cluster_colors = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252']
  features = km.feature_names_in_
  ncols = 4
  nrows = len(features) // ncols + (len(features) % ncols > 0)
  fig = plt.figure(figsize=(25,25))

  for n, feature in enumerate(features):
      ax = plt.subplot(nrows, ncols, n + 1)
      box = raw_data[[feature, 'cluster']].boxplot(by='cluster',ax=ax,return_type='both',patch_artist = True)

      for row_key, (ax,row) in box.iteritems():
          ax.set_xlabel('cluster')
          ax.set_title(feature,fontweight="bold")
          for i,box in enumerate(row['boxes']):
              box.set_facecolor(cluster_colors[i])
              
  
  fig.suptitle('Feature distributions per cluster', fontsize=18, y=1)   
  plt.tight_layout()
  plt.show() """
   
            

def create_boxplot(raw_data, km):
    cluster_colors = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252']
    features = km.feature_names_in_
    ncols = 3  # Decrease number of columns to spread plots out more vertically
    nrows = len(features) // ncols + (len(features) % ncols > 0)
    fig_width = ncols * 7  # Dynamic sizing: 7 inches per column
    fig_height = nrows * 5  # Dynamic sizing: 5 inches per row
    fig = plt.figure(figsize=(fig_width, fig_height))

    for n, feature in enumerate(features):
        ax = plt.subplot(nrows, ncols, n + 1)
        box = raw_data[[feature, 'cluster']].boxplot(by='cluster', ax=ax, return_type='both', patch_artist=True)

        for row_key, (ax, row) in box.iteritems():
            ax.set_xlabel('cluster')
            ax.set_title(feature, fontweight="bold")
            for i, box in enumerate(row['boxes']):
                box.set_facecolor(cluster_colors[i])

    fig.suptitle('Feature distributions per cluster', fontsize=18, y=1.02)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, hspace=0.4, wspace=0.3)  # Adjust the layout
    plt.show()


def cluster_comparison_bar(X_comparison, colors, raw_data, deviation=True ,title="Cluster results"):
    features = X_comparison.index
    ncols = 3
    # calculate number of rows
    nrows = len(features) // ncols + (len(features) % ncols > 0)
    # set figure size
    fig_width = ncols * 7  # Dynamic sizing: 7 inches per column
    fig_height = nrows * 5  # Dynamic sizing: 5 inches per row
    fig = plt.figure(figsize=(fig_width, fig_height))
    #fig = plt.figure(figsize=(15,15), dpi=200)
    #interate through every feature
    for n, feature in enumerate(features):
        # create chart
        ax = plt.subplot(nrows, ncols, n + 1)
        X_comparison[X_comparison.index==feature].plot(kind='bar', ax=ax, title=feature, 
                                                            color=colors[0:raw_data.cluster.nunique()],
                                                            legend=False
                                                            )
        plt.axhline(y=0)
        x_axis = ax.axes.get_xaxis()
        x_axis.set_visible(False)

    c_labels = X_comparison.columns.to_list()
    c_colors = colors[0:3]
    mpats = [mpatches.Patch(color=c, label=l) for c,l in list(zip(colors[0:raw_data.cluster.nunique()],
                                                                  X_comparison.columns.to_list()))]

    fig.legend(handles=mpats,
              ncol=ncols,
              loc="upper center",
              fancybox=True,
              bbox_to_anchor=(0.5, 0.98)
              )
    axes = fig.get_axes()
    
    fig.suptitle(title, fontsize=18, y=1)
    fig.supylabel('Deviation from overall mean in %')
    plt.tight_layout()
    #plt.subplots_adjust(top=0.93)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, hspace=0.4, wspace=0.3)
    plt.show()
    
def create_singular(X_dev_rel):
  colors = ['#9EBD6E','#81a094','#775b59','#32161f', '#946846', '#E3C16F', '#fe938c', '#E6B89C','#EAD2AC',
          '#DE9E36', '#4281A4','#37323E','#95818D']

  fig = plt.figure(figsize=(10,5), dpi=200)
  X_dev_rel.T.plot(kind='bar', 
                        ax=fig.add_subplot(), 
                        title="Cluster characteristics", 
                        color=colors,
                        xlabel="Cluster",
                        ylabel="Deviation from overall mean in %"
                        )
  plt.axhline(y=0, linewidth=1, ls='--', color='black')
  plt.legend(bbox_to_anchor=(1.04,1))
  fig.autofmt_xdate(rotation=0)
  plt.tight_layout()
  plt.show()
  
  
  
class Radar(object):
    def __init__(self, figure, title, labels, rect=None):
        if rect is None:
            rect = [0.05, 0.05, 0.9, 0.9]

        self.n = len(title)
        self.angles = np.arange(0, 360, 360.0/self.n)
        
        self.axes = [figure.add_axes(rect, projection='polar', label='axes%d' % i) for i in range(self.n)]
        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, labels=title, fontsize=14, backgroundcolor="white",zorder=999) # Feature names
        self.ax.set_yticklabels([])
        
        for ax in self.axes[1:]:
            ax.xaxis.set_visible(False)
            ax.set_yticklabels([])
            ax.set_zorder(-99)

        for ax, angle, label in zip(self.axes, self.angles, labels):
            ax.spines['polar'].set_color('black')
            ax.spines['polar'].set_zorder(-99)
                     
    def plot(self, values, *args, **kw):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        values = np.r_[values, values[0]]
        self.ax.plot(angle, values, *args, **kw)
        kw['label'] = '_noLabel'
        self.ax.fill(angle, values,*args,**kw)


#Main
#data = get_factor_loadings()
raw_data, data = get_finaldset()

km = KMeans(n_clusters=3, 
            max_iter=300, 
            tol=1e-04, 
            init='k-means++', 
            n_init=10, 
            random_state=42)

#kmeans = KMeans(n_clusters=4, random_state=42)
#km_fit = kmeans.fit_predict(data)
km_fit = km.fit(data)

#ElbowVis(data)
#get_silhouette_score(data)
#magandcardinailty(data, km_fit)

raw_data['cluster'] = km.labels_
data['cluster'] = km.labels_

#I can take a couple of boxplots as examples if not all of them are looking good.
#create_boxplot(raw_data, km)


X_mean = pd.concat([pd.DataFrame(raw_data.mean().drop('cluster'), columns=['mean']), 
                   raw_data.groupby('cluster').mean().T], axis=1)

X_dev_rel = X_mean.apply(lambda x: round((x-x['mean'])/x['mean'],2)*100, axis = 1)
X_dev_rel.drop(columns=['mean'], inplace=True)
X_mean.drop(columns=['mean'], inplace=True)

X_std_mean = pd.concat([pd.DataFrame(data.mean().drop('cluster'), columns=['mean']), 
                   data.groupby('cluster').mean().T], axis=1)

X_std_dev_rel = X_std_mean.apply(lambda x: round((x-x['mean'])/x['mean'],2)*100, axis = 1)
X_std_dev_rel.drop(columns=['mean'], inplace=True)
X_std_mean.drop(columns=['mean'], inplace=True)

cluster_colors = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252']
#cluster_comparison_bar(X_dev_rel, cluster_colors, raw_data, title="Comparison of the mean per cluster to overall mean in percent")

#Creates a singular bar chart containing all the variables on one chart.
create_singular(X_dev_rel)


fig = plt.figure(figsize=(8, 8))
no_features = len(km.feature_names_in_)
radar = Radar(fig, km.feature_names_in_, np.unique(km.labels_))

for k in range(0,km.n_clusters):
    cluster_data = X_std_mean[k].values.tolist()
    radar.plot(cluster_data,  '-', lw=2, color=cluster_colors[k], alpha=0.7, label='cluster {}'.format(k))

radar.ax.legend()
radar.ax.set_title("Cluster characteristics: Feature means per cluster", size=22, pad=60)
plt.show()