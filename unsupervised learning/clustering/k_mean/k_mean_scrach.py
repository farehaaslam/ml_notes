import random
import numpy as np
class Kmean:
    def __init__(self,n_cluster,max_itr):
        self.n_cluster=n_cluster
        self.max_itr=max_itr
        self.centroids=None

    def fit_predict(self,X):
        #  selecting n_cluster random points from dataset
        random_index=random.sample(range(0,X.shape[0]),self.n_cluster)
        self.centroids=X[random_index]
        for i in range(self.max_itr):
            cluster_grp=self.assign_cluster(X)
            old_centroid=self.centroids
            # move centroid 
            self.centroids=self.move_centroid(X,cluster_grp)
            #finish
            if(old_centroid==self.centroids).all():
                break
        return cluster_grp
            


    def assign_cluster(self,X):
        cluster_group=[]
        distance=[]
        for row in X:
            for centroid in self.centroids:
                # eucledian dist of point from all centroids
               distance.append( np.sqrt(np.dot(centroid-row,centroid-row)))

            min_dist=min(distance)
            index_pos=distance.index(min_dist)
            #print(index_pos) 
            cluster_group.append(index_pos)  
 
            distance.clear()
        return np.array(cluster_group)  

    def move_centroid(self,X,cluster_grp):
        new_centroids=[]
        cluster_type=np.unique(cluster_grp)
        for type in cluster_type:
           new_centroids.append( X[cluster_grp==type].mean(axis=0))
           
        return np.array(new_centroids)

