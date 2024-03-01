#from skopt import gp_minimize
from skopt.space import Real
from scipy.stats import pearsonr
from scipy.optimize import minimize
import numpy as np
import mcda_scratch


# Load the Topsis scores from mcda_scratch.py
df = mcda_scratch.main([])
#weights = mcda_scratch.get_weights()
weights = [0.05, 0.22, 0.17, 0.18, 0.05, 0.07, 0.16, 0.10]

# Define the objective function
def objective_function(weights):
    # Perform Topsis analysis and calculate Topsis Score
    correlation, _ = pearsonr(df['Topsis Score'], df['mktcapitalisation'])
    # Return negative correlation (to maximize)
    #return -correlation
    return -1

# Define the search space
space = [Real(0.0, 1.0, name='w{}'.format(i)) for i in range(len(weights))]
#num_variables = governance_scores.shape[1]  # Number of governance variables
initial_guess = [1 / num_variables] * num_variables

# Run Bayesian optimization
#result = gp_minimize(objective_function, space, n_calls=50, random_state=42)
result = minimize(
    objective_function, 
    initial_guess, 
    args=(governance_scores, market_caps),
    method='SLSQP', 
    bounds=bounds, 
    constraints=constraints
)

# Get the optimal weights
optimal_weights = result.x
# Normalize the optimal weights
optimal_weights_normalized = optimal_weights / np.sum(optimal_weights)

print(optimal_weights_normalized)


# Perform Topsis analysis with optimal weights
optimal_score = mcda_scratch.main(optimal_weights_normalized)

# Calculate Pearson correlation with optimal weights
optimal_correlation, _ = pearsonr(optimal_score['Topsis Score'], df['mktcapitalisation'])
print(optimal_correlation)
