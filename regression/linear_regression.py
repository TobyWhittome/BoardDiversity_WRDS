import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np


df = pd.read_excel('final_dataset.xlsx')

x = df['CEODuality']
#y = df['mktcapitalisation']
y = df['tobinsQ']

#Regression y on x, to predict mcap based off percentage INEDs
slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

x = np.array(x)

plt.scatter(x, y)
plt.plot(x, mymodel)
print(x.min(), x.max())
print(y.min(), y.max())

r_squared = r**2
print(f'R-squared value: {r_squared}')

print(f"the gradient is {slope}")


#plt.xlim(0, 3)
#plt.ylim(0, 250000)
plt.show()