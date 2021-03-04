# impoer libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,3:5].values

# create dendogram for hc
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram( sch.linkage(X, method = 'ward') )
plt.title("dendrogram")
plt.xlabel('customers')
plt.ylabel('distance')
plt.show()

# prepare clusters through aglomerative model
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering( n_clusters = 5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# visulaize our model 
c_ = ['red','blue','green','yellow','orange']
l = np.arange(1,6)
for i in range(0,5):
    plt.scatter(X[y_hc == i, 0], X[y_hc == i, 1], s = 100, c = c_[i], label = l[i] )
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
















