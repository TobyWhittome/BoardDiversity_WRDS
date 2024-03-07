import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Load data
df = pd.read_excel('final_dataset.xlsx')

X = df[['CEODuality']]
y = df[['tobinsQ']]

# Perform a t-test
t_test_result = stats.ttest_ind(df[df['CEODuality'] == 0]['tobinsQ'],
                                df[df['CEODuality'] == 1]['tobinsQ'])

# Calculate point-biserial correlation coefficient
correlation = stats.pointbiserialr(df['CEODuality'], df['tobinsQ'])

print("T-test result:", t_test_result)
print("Point-Biserial Correlation:", correlation)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Initialize and fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(report)

# Optional: Plotting the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
