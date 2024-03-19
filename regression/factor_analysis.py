import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity

df = pd.read_excel('dataset/final_dataset.xlsx')
df.drop(columns=['ticker'], inplace=True)
df.drop(columns=['mktcapitalisation'], inplace=True)
df.drop(columns=['tobinsQ'], inplace=True)
print(df)

kmo_all,kmo_model=calculate_kmo(df)
print(kmo_model)

chi_square_value,p_value=calculate_bartlett_sphericity(df)
print(chi_square_value, p_value)

""" # Test for the number of factors we must reduce to.
# We get 3 values greater than 1. So we will use n_factors=3 in our next FA.
fa = FactorAnalyzer()
fa.set_params(n_factors=8, rotation=None)
fa.fit(df)
ev, v = fa.get_eigenvalues()
print(ev) """


#df = df.iloc[1:15]
#X = df[['total_memberships', 'CEODuality','percentage_NEDs']]

fa = FactorAnalyzer(n_factors=3, rotation=None) 
fa.fit(df)

loadings = fa.loadings_
print("Factor Loadings:")
print(loadings)

# Get variance of each factors
print(fa.get_factor_variance())


# Plot the scree plot to determine the number of factors
ev, v = fa.get_eigenvalues()
plt.plot(range(1, len(ev) + 1), ev, marker='o')
plt.title('Scree Plot')
plt.xlabel('Factor Number')
plt.ylabel('Eigenvalue')
plt.show()
