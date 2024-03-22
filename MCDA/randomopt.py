import numpy as np
import MCDA as mcda
import pandas as pd

df = pd.read_excel('dataset/transformed_dataset.xlsx').dropna(inplace=True)



def generate_random_weights(n, increment):
    """Generate a random set of weights that sum to 1."""
    weights = []
    for _ in range(n):
        # Generate a single random weight and append it to the list
        weight = np.random.choice(np.arange(increment, 1.0, increment))
        weights.append(weight)
    # Convert weights to a numpy array for easy normalization
    weights_array = np.array(weights)
    # Normalize the weights so they sum to 1
    normalized_weights = weights_array / weights_array.sum()
    return normalized_weights.tolist()

def find_good_weights(n=9, increment=0.001, target_corr=0.3, attempts=1000):
    best_corr = 0
    best_weights = None
    ticker = 0
    for _ in range(attempts):
        ticker += 1
        if ticker % 100 == 0:
            print(f"Attempt {ticker}")
        weights = generate_random_weights(n, increment)
        corr = mcda.main(df, weights)
        if corr > best_corr:
            best_corr, best_weights = corr, weights
            if corr > target_corr:
                break
    return best_weights, best_corr

# Replace calculate_correlation with your actual function to compute the correlation coefficient
best_weights, best_corr = find_good_weights()
print(f"Best Weights: {best_weights}\nCorrelation Coefficient: {best_corr}")


#Best Weights: [0.025069637883008356, 0.23398328690807801, 0.20334261838440112, 0.22562674094707524, 0.01392757660167131, 0.011142061281337047, 0.038997214484679674, 0.16434540389972144, 0.08356545961002786]
#Correlation Coefficient: 0.12689609158436632

#Best Weights: [0.052367288378766134, 0.21855571496891438, 0.22405547584887614, 0.22142515542802485, 0.007651841224294595, 0.05595408895265423, 0.011477761836441893, 0.18938307030129123, 0.019129603060736487]
#Correlation Coefficient: 0.1255240540962164