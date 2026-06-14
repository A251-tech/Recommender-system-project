"'' KNN """
import csv
import array as ar
import numpy as np
import pandas as pd
df = pd.read_csv(r"C:\Users\ABHINAV CHAUHAN\Downloads\Book_2.csv")
df.head(2)
df.shape
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=2).fit(df)
neigh_dist, neigh_ind = nbrs.kneighbors(df)
sort_neigh_dist = np.sort(neigh_dist, axis=0)
import matplotlib.pyplot as plt
k_dist = sort_neigh_dist[:, 1]
plt.plot(k_dist)
plt.axhline(y=2.5, linewidth=1, linestyle='dashed', color='k')
plt.ylabel("k-NN distance")
plt.xlabel("Sorted observations (4th NN)")
plt.show()
from sklearn.cluster import DBSCAN
clusters = DBSCAN(eps=1, min_samples=4).fit(df)
# get cluster labels
clusters.labels_
# output
#array([0, 0, 1, ..., 1, 1, 1], dtype=int64)

# check unique clusters
set(clusters.labels_)
#{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, -1}
# -1 value represents noisy points could not assigned to any cluster

import seaborn as sns
import matplotlib.pyplot as plt
p = sns.scatterplot(data=df, x="vectorized1", y="vectorized2", hue=clusters.labels_, legend="full", palette="deep")
sns.move_legend(p, "upper right", bbox_to_anchor=(1.17, 1.2), title='Clusters')
plt.show()


'''

from sklearn.neighbors import
KNeighborsClassifier from sklearn.metrics
import accuracy_score
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
pred = neigh.predict(X_test)
print ("KNeighbors accuracy score :",accuracy_score(y_test,
pred))'''
