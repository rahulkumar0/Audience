# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('visitor-interests.csv')

#check null values
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(5)
df = df.drop(df[df['Interests'].isnull()].index)

X = df.drop(['IP'],axis =1)

#encoding
col = X.shape[1]
for i in range(col):
    lb = LabelEncoder()
    X.iloc[:,i] = lb.fit_transform(X.iloc[:,i])

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
X = X.values


#plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], X[y_kmeans == 0, 2], X[y_kmeans == 0, 3], cmap=plt.hot())
ax.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], X[y_kmeans == 1, 2], X[y_kmeans == 1, 3], cmap=plt.hot())
ax.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], X[y_kmeans == 2, 2], X[y_kmeans == 2, 3], cmap=plt.hot())
ax.set_xlabel('user_agent')
ax.set_ylabel('country')
ax.set_zlabel('language')
ax.set_autoscale_on(True)
plt.show()
