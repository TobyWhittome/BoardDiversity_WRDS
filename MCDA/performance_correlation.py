import pandas as pd
from scipy.stats import pearsonr
import mcda_scratch

# Load the Topsis scores from mcda_scratch.py
df = mcda_scratch.main([])

# Calculate the correlation between Topsis score and market cap
correlation, _ = pearsonr(df['Topsis Score'], df['tobinsQ'])

print(f"Correlation between Topsis score and Tobin's Q: {correlation}")


correlation_t, _ = pearsonr(df['Topsis Score'], df['mktcapitalisation'])

print(f"Correlation between Topsis score and market cap: {correlation_t}")