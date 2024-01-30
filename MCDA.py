import create_dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load your dataset
df = create_dataset.main()
print(df)


# Extract the variables (features) for normalization
variables = df[['total_memberships', 'high_voting_power']]

# Normalize the variables using Min-Max scaling
scaler = MinMaxScaler()
normalized_variables = scaler.fit_transform(variables)

# Convert the normalized variables back to a DataFrame
normalized_data = pd.DataFrame(normalized_variables, columns=variables.columns)

# Assign equal weights to each variable
weights = pd.Series(1, index=variables.columns)

# Calculate the weighted sum
soundness_score = (normalized_data * weights).sum(axis=1)

# Add the soundness score to the original DataFrame
df['soundness_score'] = soundness_score

print(df)
