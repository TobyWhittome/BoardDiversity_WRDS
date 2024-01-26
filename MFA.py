import pandas as pd
from prince import MFA
import create_dataset

# Load your dataset
df = create_dataset.main()
print(df)


# Assuming df is your Pandas DataFrame with mixed data
mfa_model = MFA(n_components=2)
mfa_model.fit(df)


# Eigenvalues
print(mfa_model.eigenvalues_)

# Eigenvectors
print(mfa_model.eigenvectors_)

# Coordinates
print(mfa_model.row_coordinates(df))  # Row coordinates
print(mfa_model.column_coordinates(df))  # Column coordinates


""" # Visualization of row and column coordinates
mfa_model.plot_row_coordinates(df, color_labels=df['Category'])
mfa_model.plot_column_coordinates(df, color_labels=df['Category'])
 """