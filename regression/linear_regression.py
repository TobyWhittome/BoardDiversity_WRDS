import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd


df = pd.read_excel('final_dataset.xlsx')

x = df['total_share_%']
y = df['mktcapitalisation']

#Regression y on x, to predict mcap based off percentage INEDs
slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
print(x.min(), x.max())
print(y.min(), y.max())


plt.xlim(0, 3)
plt.ylim(0, 250000)
plt.show()