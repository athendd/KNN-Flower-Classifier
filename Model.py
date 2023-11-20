import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


iris = sns.load_dataset("iris")
df = pd.DataFrame(iris)
print(df.head())
print(df.info())
print(df.describe())
print(df.dtypes)

sepal_length is float, sepal_width is float, petal_length is float, petal_width is float, species is object
petal_length and petal_width might have outliers
all appear to have no null values
print(df.isnull().sum())
#no null values in the dataset

df['species'] = df['species'].astype('category')
target = df['species']
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
correlation_matrix = df[columns].corr()
sns.heatmap(data = correlation_matrix, annot = True)
plt.savefig('/workspaces/One/heatmap.png')  # Save the plot as an image
plt.show()
plt.clf()


plt.boxplot(df['sepal_length'])
plt.show()
plt.clf()
plt.boxplot(df['sepal_width'])
plt.savefig('sepal_width.png')
plt.show()
plt.clf()
plt.boxplot(df['petal_length'])
plt.savefig('petal_length.png')
plt.show()
plt.clf()
plt.boxplot(df['petal_width'])
plt.savefig('petal_width.png')
plt.clf()

#sepal_width is the only column with outliers
plt.hist(df['sepal_width'])
plt.savefig('sepal_width_hist.png')
plt.show()
plt.clf()
upper_limit = 0.095
lower_limit = 0.009
df['sepal_width'] = winsorize(df['sepal_width'], limits = (lower_limit, upper_limit))
plt.boxplot(df['sepal_width'])
plt.savefig('updated_sepal_width.png')
plt.show()
plt.clf()
scaler = StandardScaler()
x = scaler.fit_transform(df[columns])
y = df['species']


#used a for loop to figure out that the best value for k was 1
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = 0.2, random_state = 42)
accuracies = []
classifier = KNeighborsClassifier(n_neighbors = 1)
classifier.fit(x_train, y_train)
print(classifier.score(x_val, y_val))
y_pred = classifier.predict(x_val)
accuracy = accuracy_score(y_val, y_pred)
print(accuracy)

#got it to an accuracy of 0.96666666666667

