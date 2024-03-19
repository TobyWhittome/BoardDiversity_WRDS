import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


df = pd.read_excel('dataset/transformed_dataset.xlsx')
y = df['tobinsQ']
df.drop(columns=['ticker'], inplace=True)
df.drop(columns=['mktcapitalisation'], inplace=True)
df.drop(columns=['tobinsQ'], inplace=True)


for col in df.columns:

  x = df[col]
  print(f"Variable in question: {x.name}")

  slope, intercept, r, p, std_err = stats.linregress(x, y)
  print(f"Gradient: {slope}")

  correlation, _ = pearsonr(x, y)
  print(f"Correlation between Topsis score and Tobin's Q: {correlation}")

  correlationspear, _ = stats.spearmanr(x, y)
  print(f"Spearman's rank correlation: {correlationspear} \n")

  #check for a basic correlation
  
      


def myfunc(x):
  return slope * x + intercept

def visualize():
  mymodel = list(map(myfunc, x))

  x = np.array(x)

  plt.scatter(x, y)
  plt.plot(x, mymodel)
  print(x.min(), x.max())
  print(y.min(), y.max())

  r_squared = r**2
  print(f'R-squared value: {r_squared}')

  print(f"the gradient is {slope}")
  plt.show()