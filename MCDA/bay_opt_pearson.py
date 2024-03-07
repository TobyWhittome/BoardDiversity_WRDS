import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import minimize
import mcda_scratch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

df = pd.read_excel("transformed_dataset.xlsx")
no_mcaptobn_df = df.copy()
no_mcaptobn_df.drop(columns=['mktcapitalisation', 'tobinsQ'], inplace=True)
print(no_mcaptobn_df)

# Assuming `criteria_scores` is a DataFrame where rows are items and columns are the scores for each criterion
# `market_caps` is a Series or array with the market capitalization for each item


def get_criteria_scores(df):
  # Assuming `df` is your raw data DataFrame
  scaler = MinMaxScaler()
  criteria_scores = scaler.fit_transform(df[['high_voting_power', 'percentage_INEDs', 'num_directors_>4.5', 'total_share_%', 'total_memberships', 'CEODuality', 'dualclass', 'boardsize_mean']])
  # Now, `criteria_scores` is a numpy array with normalized scores. If you need it back as a DataFrame:
  criteria_scores_df = pd.DataFrame(criteria_scores, columns=['high_voting_power', 'percentage_INEDs', 'num_directors_>4.5', 'total_share_%', 'total_memberships', 'CEODuality', 'dualclass', 'boardsize_mean'])
  return criteria_scores_df


# Redefined Objective Function
def objective(weights, criteria_scores, market_caps):
    # Calculate the weighted sum of criteria scores to get the composite score for each item
    composite_scores = np.dot(criteria_scores, weights)
    # Calculate the Pearson correlation coefficient between composite scores and market caps
    
    corr, _ = pearsonr(composite_scores, market_caps)
    return -corr  # Negate because we want to maximize the correlation


criteria_scores = get_criteria_scores(no_mcaptobn_df)
# Example of setup for optimization (you'll need to adjust according to your actual data structure)
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # Weights sum to 1
bounds = [(0, 1) for _ in range(criteria_scores.shape[1])]  # Bounds for each weight
#initial_guess = [1 / criteria_scores.shape[1]] * criteria_scores.shape[1]
initial_guess = [0.05, 0.22, 0.17, 0.18, 0.05, 0.07, 0.16, 0.10]

# Initial guess for weights


# Perform the optimization
result = minimize(
    objective, 
    initial_guess, 
    args=(criteria_scores.values, df['tobinsQ'].values),  # Assuming `criteria_scores` and `market_caps` are correctly aligned
    method='SLSQP', 
    bounds=bounds, 
    constraints=constraints
)

optimized_weights = result.x
print(optimized_weights)


optimal_weights_normalized = optimized_weights / np.sum(optimized_weights)
print(optimal_weights_normalized)




# Perform Topsis analysis with optimal weights
#optimal_score = mcda_scratch.main(optimal_weights_normalized)


#correlation, _ = pearsonr(df['Topsis Score'], df['tobinsQ'])
#print(f"Correlation between Topsis score and Tobin's Q: {correlation}")



