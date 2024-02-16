from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import create_dataset
import pandas as pd
import matplotlib as plt


# Load your dataset
df = create_dataset.main()

# Let's assume that the last column is your target variable
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Create an LDA object
lda = LinearDiscriminantAnalysis(n_components=2)  # Change n_components to the number of linear discriminants you want

# Fit the model

X_r = lda.fit(X, y).transform(X)

# Now X_r contains the transformed data

# Visualizing the transformed data
plt.figure()
plt.title('LDA of dataset')
plt.show()