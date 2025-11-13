from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from k_mean_scrach import Kmean
centers=[(5,5),(-6,-4),[8,-3]]
std=[2,2,1]
X,y=make_blobs(n_features=3,centers=centers,n_samples=500,cluster_std=std)
# plt.scatter(X[:,0],X[:,1])
# plt.show()
k=Kmean(3,200)
y_mean=k.fit_predict(X)
plt.scatter(X[y_mean == 0, 0], X[y_mean == 0, 1], color='red')
plt.scatter(X[y_mean == 1, 0], X[y_mean == 1, 1], color='blue')
plt.scatter(X[y_mean == 2, 0], X[y_mean == 2, 1], color='pink')

plt.show()

