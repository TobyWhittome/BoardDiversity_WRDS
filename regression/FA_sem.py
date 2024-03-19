import semopy as sem
import pandas as pd

data = pd.read_excel('dataset/final_dataset.xlsx')
data.drop(columns=['ticker'], inplace=True)
data = data.rename(columns={'high_voting_power': 'y1', 'percentage_INEDs': 'y2', 'num_directors_>4.5': 'y3', 'total_share_%': 'y4', 'total_memberships':'y5', 'boardsize':'y6', 'CEODuality': 'x1', 'dualclass': 'x2', 'mktcapitalisation': 'x3'})

# Define the SEM model
model_desc = """
# Measurement model
eta1 =~ y1 + y2 + y3
eta2 =~ y4 + y5 + y6
eta3 =~ x1 + x2 + x3

# Structural model
eta2 ~ eta1
eta3 ~ eta2
"""

# Initialize and estimate the model
model = sem.Model(model_desc)
results = model.fit(data, obj="DWLS", solver="SLSQP")

# Print out estimated parameters
print("Estimated parameters:")
print(results)


# Assess model fit
print("\nModel fit indices:")
model.inspect(mode='list', what="names", std_est=True)
print(model.inspect())

# Calculate and display goodness-of-fit measures
calc_stuff = sem.calc_stats(model)
print(calc_stuff)

#Visualise
#g = sem.semplot(model, "model.png")
