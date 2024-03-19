import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def get_criteria_scores(df):
  scaler = MinMaxScaler()
  criteria_scores = scaler.fit_transform(df[['high_voting_power', 'percentage_INEDs', 'num_directors_>4.5', 'total_share_%', 'total_memberships', 'CEODuality', 'dualclass', 'boardsize_mean']])
  # Now, `criteria_scores` is a numpy array with normalized scores. If you need it back as a DataFrame:
  criteria_scores_df = pd.DataFrame(criteria_scores, columns=['high_voting_power', 'percentage_INEDs', 'num_directors_>4.5', 'total_share_%', 'total_memberships', 'CEODuality', 'dualclass', 'boardsize_mean'])
  return criteria_scores_df


def objective(weights, criteria_scores, market_caps):
    # Calculate the weighted sum of criteria scores to get the composite score for each item
    composite_scores = np.dot(criteria_scores, weights)
    # Calculate the Pearson correlation coefficient between composite scores and market caps
    
    corr, _ = pearsonr(composite_scores, market_caps)
    print(corr)
    print("CORRRR")
    return -corr  # Negate because we want to maximize the correlation
   

df = pd.read_excel("dataset/transformed_dataset.xlsx")
no_mcaptobn_df = df.copy()
no_mcaptobn_df.drop(columns=['ticker','mktcapitalisation', 'tobinsQ'], inplace=True)
print(no_mcaptobn_df)


criteria_scores = get_criteria_scores(no_mcaptobn_df)
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # Weights sum to 1
bounds = [(0, 1) for _ in range(criteria_scores.shape[1])]  # Bounds for each weight
initial_guess = [0.05, 0.22, 0.17, 0.18, 0.05, 0.07, 0.16, 0.10]


result = minimize(
    objective, 
    initial_guess, 
    args=(criteria_scores.values, df['tobinsQ'].values),
    method='SLSQP', 
    bounds=bounds, 
    constraints=constraints)

optimized_weights = result.x
print(optimized_weights)


optimal_weights_normalized = optimized_weights / np.sum(optimized_weights)
print(optimal_weights_normalized)


# Perform Topsis analysis with optimal weights
#optimal_score = mcda_scratch.main(optimal_weights_normalized)


#correlation, _ = pearsonr(df['Topsis Score'], df['tobinsQ'])
#print(f"Correlation between Topsis score and Tobin's Q: {correlation}")



