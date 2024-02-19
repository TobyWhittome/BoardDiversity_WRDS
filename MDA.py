from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import create_dataset
import pandas as pd
import matplotlib as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


df = create_dataset.main()


X = df.iloc[:, 1:5].values.astype(int)
print(X)
# This data has two possible values, 0 or 1. Therefore we have to reduce to 1 dimension, it cannot be 2.
y = df.iloc[:, 6].values.astype(int)
print(y)


#This one manages to reduce to 2 dimensions.
""" url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
cls = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(url, names=cls)
 
# divide the dataset into class and target variable
X = dataset.iloc[:, 0:4].values
print(X)
y = dataset.iloc[:, 4].values
print(y) """

sc = StandardScaler()
X = sc.fit_transform(X)
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

 
# apply Linear Discriminant Analysis
n_components_assign = min(X_train.shape[1], len(set(y_train)) - 1)
lda = LinearDiscriminantAnalysis(n_components=n_components_assign)

X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
 
 
#Since we have reduced it to 1 dimensions, a box plot is good for visualisation
plt.boxplot([X_train[y_train==0][:,0], X_train[y_train==1][:,0]], labels=['Class 0', 'Class 1'])
plt.xlabel('Target Class')
plt.ylabel('LDA Component')
plt.title('Boxplot of LDA Component by Target Class')
plt.show()


# only useful for 2 dimensional data
""" # plot the scatterplot
plt.scatter(
    X_train[:,0],X_train[:,1],c=y_train,cmap='rainbow',
  alpha=0.7,edgecolors='b'
) """
 
 
# classify using random forest classifier
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
 
# print the accuracy and confusion matrix
print('Accuracy : ' + str(accuracy_score(y_test, y_pred)))
conf_m = confusion_matrix(y_test, y_pred)
print(conf_m)