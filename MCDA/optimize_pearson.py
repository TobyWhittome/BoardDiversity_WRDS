import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import minimize
import MCDA.mcda_scratch as mcda_scratch

df = mcda_scratch.main([])

# Objective function to be minimized
def objective(weights, market_caps):
    # Calculate the weighted sum of governance scores
    composite_score = np.dot(df['Topsis Score'], weights)
    # Calculate the negative Pearson correlation coefficient between composite score and market cap
    corr, _ = pearsonr(composite_score, market_caps)
    return -corr  # We negate because we want to maximize correlation


# Constraint for weights to sum to 1
constraints = ({'type': 'eq', 'fun': lambda weights: 1 - sum(weights)})

# Bounds for each weight to be between 0 and 1
bounds = [(0, 1) for _ in range(8)]  # Assuming 'num_variables' is the number of governance variables


# Initial guess (equal distribution of weights)
  # Number of governance variables
initial_guess = [1 / 8] * 8

# Perform the optimization
result = minimize(
    objective, 
    initial_guess, 
    args=(df['Topsis Score'], df['mktcapitalisation']),
    method='SLSQP', 
    bounds=bounds, 
    constraints=constraints
)

optimized_weights = result.x
