import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import textwrap

def create_heatmap(df, varifa):
    factors = 4

    # Prepare the plot
    fig, ax = plt.subplots(figsize=(6.6, 8))  # Adjust the figsize as needed

    # Use varifa.loadings_ to get the factor loading matrix
    factor_matrix = varifa.loadings_

    # Plot the data as a heatmap
    im = ax.imshow(factor_matrix, cmap="RdBu_r", vmax=1, vmin=-1)

    # Add the corresponding value to the center of each cell
    for (i, j), z in np.ndenumerate(factor_matrix):
        ax.text(j, i, '{:0.2f}'.format(z), ha="center", va="center")

    # Set y-axis labels to variable names from df.columns
    
    # Wrap y-axis labels
    ax.set_yticks(np.arange(len(df.columns)))
    wrapped_labels = [textwrap.fill(label, width=13) for label in df.columns]

    ax.set_yticklabels(wrapped_labels, fontweight='bold')


    # Set x-axis labels to factor numbers
    ax.set_xticks(np.arange(factors))
    ax.set_xticklabels(["Factor {}".format(i+1) for i in range(factors)], fontweight='bold')

    #ax.set_title("Factor Analysis Varimax Loadings", fontweight='bold')

    # Adjust layout
    plt.tight_layout()

    # Add a colorbar
    #Make the loadings label bold
    
    cb = fig.colorbar(im, ax=ax, location='right', shrink=0.6)
    cb.set_label("Loadings", weight='bold')

    # Show the plot
    plt.show()


df = pd.read_excel('dataset/final_dataset.xlsx')
df.rename(columns={'Number Directors\'Own>4.5' : 'Number Directors\' Own>4.5'}, inplace=True)
df.drop(columns=['ticker'], inplace=True)
#df.drop(columns=['mktcapitalisation'], inplace=True)
#create a 1D array of Tobin's Q
tobins_q = df['tobinsQ']
df.drop(columns=['tobinsQ'], inplace=True)
#df.drop(columns=['genderratio'], inplace=True)

kmo_all,kmo_model=calculate_kmo(df)
print(kmo_model)

chi_square_value,p_value=calculate_bartlett_sphericity(df)
print(chi_square_value, p_value)

# Test for the number of factors we must reduce to.
# We get 4 values greater than 1. So we will use n_factors=4 in our next FA.
ta = FactorAnalyzer()
ta.set_params(n_factors=9, rotation=None)
ta.fit(df)
ev, v = ta.get_eigenvalues()
print(ev)


nonefa = FactorAnalyzer(n_factors=4, rotation=None).fit(df)
varifa = FactorAnalyzer(n_factors=4, rotation="varimax").fit(df)

create_heatmap(df, varifa)

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
#Create scatterplot for factor scores vs tobins Q, using matplotlib
print(factor_scores.shape)


plt.figure(figsize=(12, 10))  # Adjust the figure size as needed for a 2x2 layout

# Plot for Factor 1
plt.subplot(2, 2, 1)  # (rows, columns, panel number)
sns.scatterplot(x=factor_scores[:, 0], y=tobins_q, color='b')
#plt.title('Scatter plot of Factor 1 loadings vs Tobin\'s Q', fontweight='bold')
plt.xlabel('Factor 1 loadings', fontweight='bold', fontsize=16)
plt.ylabel('Tobins Q', fontweight='bold', fontsize=16)
plt.yticks(fontsize=14, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')
plt.ylim(-0.5, 15)


# Plot for Factor 2
plt.subplot(2, 2, 2)  # (rows, columns, panel number)
sns.scatterplot(x=factor_scores[:, 1], y=tobins_q, color='b')
#plt.title('Scatter plot of Factor 2 loadings vs Tobin\'s Q', fontweight='bold')
plt.xlabel('Factor 2 loadings', fontweight='bold', fontsize=16)
plt.ylabel('Tobins Q', fontweight='bold', fontsize=16)
plt.yticks(fontsize=14, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')
plt.ylim(-0.5, 15)


# Plot for Factor 3
plt.subplot(2, 2, 3)  # (rows, columns, panel number)
sns.scatterplot(x=factor_scores[:, 2], y=tobins_q, color='b')
#plt.title('Scatter plot of Factor 3 loadings vs Tobin\'s Q', fontweight='bold')
plt.xlabel('Factor 3 loadings', fontweight='bold', fontsize=16)
plt.ylabel('Tobins Q', fontweight='bold', fontsize=16)
plt.yticks(fontsize=14, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')
plt.ylim(-0.5, 15)


# Plot for Factor 4
plt.subplot(2, 2, 4)  # (rows, columns, panel number)
sns.scatterplot(x=factor_scores[:, 3], y=tobins_q, color='b')
#plt.title('Scatter plot of Factor 4 loadings vs Tobin\'s Q', fontweight='bold')
plt.xlabel('Factor 4 loadings', fontweight='bold', fontsize=16)
plt.ylabel('Tobins Q', fontweight='bold', fontsize=16)
plt.yticks(fontsize=14, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')
plt.ylim(-0.5, 15)


plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=4, hspace=4)

plt.tight_layout()  # Adjusts the subplots to fit into the figure area.
plt.show()


""" sns.scatterplot(x=factor_scores[:, 2], y=tobins_q, data=df, color='b')
plt.title(f'Scatter plot of Factor 3 vs vs Tobin\'s Q', fontweight='bold')
plt.xlabel('Factor 3 loadings')
plt.ylabel('Tobins Q')
plt.ylim(-0.5, 15)
plt.grid(True)
plt.show()


sns.scatterplot(x=factor_scores[:, 3], y=tobins_q, data=df, color='b')
#make the title bold

plt.title('Scatter plot of Factor 4 loadings vs Tobin\'s Q', fontweight='bold')
plt.xlabel('Factor 4 loadings')
plt.ylabel('Tobins Q')
plt.ylim(-0.5, 15)
plt.grid(True)
plt.show() """