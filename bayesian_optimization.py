from skopt import gp_minimize
from skopt.space import Real
from scipy.stats import pearsonr
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
    return -correlation

# Define the search space
space = [Real(0.0, 1.0, name='w{}'.format(i)) for i in range(len(weights))]

# Run Bayesian optimization
result = gp_minimize(objective_function, space, n_calls=50, random_state=42)

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
