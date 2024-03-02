#Factor analysis method:

import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import create_dataset

# Load your dataset
df = create_dataset.main()
print(df)
df = df.iloc[1:15]
X = df[['total_memberships', 'CEODuality','percentage_NEDs']]


# Drop any rows with missing values
#data = df.dropna()

# Extract the features (variables) you want to include in the factor analysis
# Assume 'features' is a list of column names in your dataset
#features = ['var1', 'var2', 'var3', '...']

# Subset the dataset with selected features
#subset_data = data[features]

# Initialize the factor analyzer with the desired number of factors
# You can adjust the number of factors based on your analysis
fa = FactorAnalyzer(n_factors=3, rotation=None) 

# Fit the model to the data
fa.fit(X)

# Get the factor loadings
loadings = fa.loadings_

# Print the factor loadings
print("Factor Loadings:")
print(loadings)

# Plot the scree plot to determine the number of factors
ev, v = fa.get_eigenvalues()
plt.plot(range(1, len(ev) + 1), ev, marker='o')
plt.title('Scree Plot')
plt.xlabel('Factor Number')
plt.ylabel('Eigenvalue')
plt.show()
