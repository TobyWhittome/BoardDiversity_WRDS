import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import numpy as np
from sklearn.linear_model import LinearRegression


df = pd.read_excel('dataset/final_dataset.xlsx')
df.drop(columns=['ticker'], inplace=True)
df.drop(columns=['mktcapitalisation'], inplace=True)
#create a 1D array of Tobin's Q
tobins_q = df['tobinsQ']
df.drop(columns=['tobinsQ'], inplace=True)

kmo_all,kmo_model=calculate_kmo(df)
print(kmo_model)

chi_square_value,p_value=calculate_bartlett_sphericity(df)
print(chi_square_value, p_value)

""" # Test for the number of factors we must reduce to.
# We get 3 values greater than 1. So we will use n_factors=3 in our next FA.
ta = FactorAnalyzer()
ta.set_params(n_factors=9, rotation=None)
ta.fit(df)
ev, v = ta.get_eigenvalues()
print(ev) """


nonefa = FactorAnalyzer(n_factors=3, rotation=None).fit(df)
varifa = FactorAnalyzer(n_factors=3, rotation="varimax").fit(df)

#create_heatmap(df, varifa, nonefa)

loadings = varifa.loadings_
print("Factor Loadings:")
print(loadings)

# Get variance of each factors
print(f"Variances of each factor: {varifa.get_factor_variance()}")


#Regression using factor scores

# Get the factor scores for the observations
factor_scores = varifa.transform(df)  # This provides the factor scores
# Create a linear regression model to predict Tobin's Q
regression_model = LinearRegression()
# Fit the regression model using the factor scores as predictors and Tobin's Q as the response
regression_model.fit(factor_scores, tobins_q)
# Output the regression coefficients to understand the relationship with Tobin's Q
print(f"Factor correlations with Tobin's Q: {regression_model.coef_}")

#If want to scatterplot them -- however shows no correlation
""" #Create scatterplot for factor scores vs tobins Q, using matplotlib
print(factor_scores.shape)
plt.scatter(factor_scores[:, 0], tobins_q)
plt.title("Factor 1 vs Tobin's Q")
plt.xlabel("Factor 1")
plt.ylabel("Tobin's Q")
plt.show()

#create another one for the next factor scores row
 """

#Visualisation
def create_heatmap(df, varifa, nonefa):
    factors = 3

    # a list of tuples containing titles for and instances of FactorAnalyzer class
    fas = [
        ("FA no rotation", nonefa),
        ("FA varimax", varifa),
    ]  

    # Let's prepare some plots on one canvas (subplots)
    fig, axes = plt.subplots(ncols=len(fas), figsize=(10, 8), squeeze=False)  # Ensure axes is always 2D
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for ax, (title, fa) in zip(axes, fas):
        # Transpose the component (loading) matrix
        factor_matrix = fa.loadings_  # Use loadings_ instead of components_
        # Plot the data as a heat map
        im = ax.imshow(factor_matrix, cmap="RdBu_r", vmax=1, vmin=-1)
        # Add the corresponding value to the center of each cell
        for (i, j), z in np.ndenumerate(factor_matrix):
            ax.text(j, i, '{:0.2f}'.format(z), ha="center", va="center")
        # Tell matplotlib about the metadata of the plot
        ax.set_yticks(np.arange(len(df.columns)))
        if ax is axes[0]:  # Check if it's the first subplot
            ax.set_yticklabels(df.columns)
        else:
            ax.set_yticklabels([])
        ax.set_title(title)
        ax.set_xticks(np.arange(factors))
        ax.set_xticklabels(["Factor {}".format(i+1) for i in range(factors)])
    # Adjust layout
    plt.tight_layout()

    # Add a colorbar
    cb = fig.colorbar(im, ax=axes, location='right', shrink=0.6, label="Loadings")
    # Show the plot
    plt.show()