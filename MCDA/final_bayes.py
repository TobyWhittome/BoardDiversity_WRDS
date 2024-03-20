import numpy as np
from scipy.optimize import minimize
from scipy import stats
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from MCDA import main


def compute_topsis_score(weights, data):

    df = main([weights])
    topsis_scores = df['Topsis_Score']
    print(topsis_scores)
    return topsis_scores

# The objective function to minimize (negative Spearman correlation)
def objective(weights, df, tobin_q_column_name):
    topsis_scores = compute_topsis_score(weights, df)
    correlation, _ = stats.spearmanr(topsis_scores, df[tobin_q_column_name])
    return -correlation  # Minimize negative correlation to maximize positive correlation

def get_criteria_scores(df):
  scaler = MinMaxScaler()
  criteria_scores = scaler.fit_transform(df[['high_voting_power', 'percentage_INEDs', 'num_directors_>4.5', 'total_share_%', 'total_memberships', 'CEODuality', 'dualclass', 'boardsize_mean']])
  # Now, `criteria_scores` is a numpy array with normalized scores. If you need it back as a DataFrame:
  criteria_scores_df = pd.DataFrame(criteria_scores, columns=['high_voting_power', 'percentage_INEDs', 'num_directors_>4.5', 'total_share_%', 'total_memberships', 'CEODuality', 'dualclass', 'boardsize_mean'])
  return criteria_scores_df

df = pd.read_excel("dataset/transformed_dataset.xlsx")
no_mcaptobn_df = df.copy()
no_mcaptobn_df.drop(columns=['ticker','mktcapitalisation', 'tobinsQ'], inplace=True)
print(no_mcaptobn_df)
criteria_scores = get_criteria_scores(no_mcaptobn_df)


# Initial guess (can be your current weights or any other guess)
#initial_weights = np.array([0.303, 0.04, 0.277, 0.032, 0.101, 0.121, 0.024, 0.099])
initial_weights = [0.303, 0.04, 0.277, 0.032, 0.101, 0.121, 0.024, 0.099]

# Constraint: sum of weights = 1
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

# Bounds for each weight (between 0 and 1)
bounds = [(0, 1) for _ in range(criteria_scores.shape[1])]

# Optimization (this is a placeholder, choose your specific optimization method)
result = minimize(objective, initial_weights, args=(criteria_scores.values, df['tobinsQ'].values), method='SLSQP', bounds=bounds, constraints=constraints)

# Optimized weights
optimized_weights = result.x
