import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats # Make sure to import your script here correctly
import MCDA as mcda

# Load your dataset
df = pd.read_excel('dataset/transformed_dataset.xlsx')
df.dropna(inplace=True)

# Original weights from your script (as a reference, you can choose any set of weights you used)
#original_weights = [7.40128150e-04, 4.36675609e-02, 3.56649252e-02, 2.00759761e-02, 1.12879591e-04, 1.58934464e-02, 5.14730934e-03, 2.24856145e-02, 8.56212160e-01]

original_weights = [2, 30, 10, 13, 3, 3, 4, 16, 10]

# Assuming we focus on the first three weights as key weights
key_weights_indices = [0, 1, 2,3,4,5,6,7,8]
#critical_values = [0.01, 0.5, 0.99]  # Minimum, midpoint, and maximum values

#critical_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]  
#critical_values =[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4]
critical_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]

# Initialize dictionary to store Spearman correlations for critical values
correlations = {i: [] for i in key_weights_indices}
highest = 0.14

for i in key_weights_indices:
    for value in critical_values:
        temp_weights = original_weights.copy()
        # Adjust the current key weight
        temp_weights[i] = value
        # Normalize the weights
        temp_weights = temp_weights / np.sum(temp_weights)
        # Run the MCDA with adjusted weights
        _, spearman_corr = mcda.main(df.copy(), temp_weights)
        if spearman_corr > highest:
            highest = spearman_corr
        correlations[i].append(spearman_corr)

# Print or plot the simplified sensitivity analysis results
for i, values in correlations.items():
    print(f"Weight {i+1} - Critical Values: {critical_values}, Correlations: {values}")

# Calculate the difference in the largest and smallest values of each results in the sensitivity analysis
differences = {i: max(values) - min(values) for i, values in correlations.items()}
print(f"Differences: {differences}")

""" # Define the range and step size for sensitivity analysis
weight_range = np.linspace(0.01, 0.99, 10)  # Vary weights from 0.01 to 0.99 in 10 steps

# Store the correlation results for each weight variation
correlation_results = []

# Perform sensitivity analysis by varying each weight one at a time
for i, _ in enumerate(original_weights):
    temp_correlations = []
    for w in weight_range:
        # Copy original weights and modify only the current weight under analysis
        temp_weights = original_weights.copy()
        temp_weights[i] = w
        # Normalize the modified weights so they sum to 1
        temp_weights = temp_weights / np.sum(temp_weights)
        
        # Run the MCDA process with the modified weights
        _, spearman_corr = mcda.main(df.copy(), temp_weights)
        temp_correlations.append(spearman_corr)
    
    correlation_results.append(temp_correlations)

# Plotting the sensitivity analysis results
for i, corr in enumerate(correlation_results):
    plt.plot(weight_range, corr, label=f'Weight {i+1}')

plt.xlabel('Weight Value')
plt.ylabel("Spearman's Rank Correlation")
plt.title('Sensitivity Analysis of Weights')
plt.legend()
plt.grid(True)
plt.show()
 """