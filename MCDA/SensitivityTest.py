import pandas as pd
import numpy as np
import MCDA as mcda

df = pd.read_excel('dataset/transformed_dataset.xlsx')
df.dropna(inplace=True)

original_weights = [2, 30, 10, 13, 3, 3, 4, 16, 10]
key_weights_indices = [0, 1, 2,3,4,5,6,7,8]
critical_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]

correlations = {i: [] for i in key_weights_indices}
highest = 0.14

for i in key_weights_indices:
    for value in critical_values:
        temp_weights = original_weights.copy()
        temp_weights[i] = value
        temp_weights = temp_weights / np.sum(temp_weights)
        _, spearman_corr = mcda.main(df.copy(), temp_weights)
        if spearman_corr > highest:
            highest = spearman_corr
        correlations[i].append(spearman_corr)

for i, values in correlations.items():
    print(f"Weight {i+1} - Critical Values: {critical_values}, Correlations: {values}")

# Calculate the difference in the largest and smallest values of each results in the sensitivity analysis
differences = {i: max(values) - min(values) for i, values in correlations.items()}
print(f"Differences: {differences}")