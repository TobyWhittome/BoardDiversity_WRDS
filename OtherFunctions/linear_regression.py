import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def myfunc(x):
  return slope * x + intercept


df = pd.read_excel('Data/mcda_dataset.xlsx')
y = df['tobinsQ']
df.drop(columns=['ticker'], inplace=True)
#df.drop(columns=['mktcapitalisation'], inplace=True)
df.drop(columns=['tobinsQ'], inplace=True)

spearmans = []

for col in df.columns:

  x = df[col]
  variable = x.name
  print(f"Variable in question: {variable}")

  slope, intercept, r, p, std_err = stats.linregress(x, y)
  print(f"Gradient: {slope}")

  correlation, _ = pearsonr(x, y)
  print(f"Correlation between Topsis score and Tobin's Q: {correlation}")

  correlationspear, _ = stats.spearmanr(x, y)
  print(f"Spearman's rank correlation: {correlationspear} \n")
  spearmans.append(correlationspear)

  mymodel = list(map(myfunc, x))

  x = np.array(x)

  plt.scatter(x, y)
  plt.plot(x, mymodel)
  #plt.title(variable)
  plt.xlabel(variable, fontweight='bold', fontsize=12)
  plt.ylabel('Tobin\'s Q', fontweight='bold', fontsize=12)
  
  #print(x.min(), x.max())
  #print(y.min(), y.max())

  r_squared = r**2
  #print(f'R-squared value: {r_squared}')

  print(f"the gradient is {slope}")
  plt.show()
