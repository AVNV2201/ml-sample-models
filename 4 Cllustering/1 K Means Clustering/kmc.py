import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

# elblow method to find the value of k 
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    clusterer = KMeans( n_clusters = i, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0 )
    clusterer.fit(X)
    wcss.append( clusterer.inertia_ )

# plotting the graph for elbow method check
plt.plot( range(1,11), wcss )
plt.title('KMeans elbow method ')
plt.xlabel('k')
plt.ylabel('wcss')
plt.show()

# prepare a model with k = 5
clusterer = KMeans( n_clusters = 5, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0 )
y_kmeans = clusterer.fit_predict(X)

# visualize the model
c_ = ['red','blue','green','yellow','orange']
l = np.arange(1,6)
for i in range(0,5):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s = 100, c = c_[i], label = l[i] )
plt.scatter(clusterer.cluster_centers_[:, 0], clusterer.cluster_centers_[:, 1], s = 300, c = 'black', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()











