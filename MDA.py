from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import create_dataset
import pandas as pd


# Load your dataset
df = create_dataset.main()

# Let's assume that the last column is your target variable
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Create an LDA object
lda = LDA(n_components=2)  # Change n_components to the number of linear discriminants you want

# Fit the model

X_r = lda.fit(X, y).transform(X)

# Now X_r contains the transformed data

# Visualizing the transformed data
plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of dataset')

plt.show()