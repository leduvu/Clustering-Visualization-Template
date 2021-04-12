#X = [[1.0 1.0][1.0 1.0][1.0 1.0][1.0 1.0]]		[[numpy.ndarray][numpy.ndarray]] is also numpy.ndarray
#X = StandardScaler().fit_transform(X)


from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn import metrics

from sklearn.manifold import TSNE

import numpy as np
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt




# X = np.array([[vec], [vec], [vec]])
# sentences = ['s','s','s','s']

def clustering(X, sentences):
	db = DBSCAN(eps=0.3, min_samples=2).fit(X)
	labels = db.labels_
	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	n_noise_ = list(labels).count(-1)

	print('Estimated number of clusters: %d' % n_clusters_)
	print('Estimated number of noise points: %d' % n_noise_)

	tagged = list(zip(labels, sentences))

	for element in tagged:
		print(element)

	tsne = TSNE(n_components=2, random_state=0)
	np.set_printoptions(suppress=True)
	Y = tsne.fit_transform(X)

	x_coords = Y[:, 0]
	y_coords = Y[:, 1]
	plt.scatter(x_coords, y_coords, c=labels)
	plt.show()


def k_means(X, sentences):	
	NUM_CLUSTERS = 3 
	kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
	kmeans.fit(X)
	  
	labels = kmeans.labels_
	centroids = kmeans.cluster_centers_

	tagged = list(zip(labels, sentences))

	for element in tagged:
		print(element)
	 
	model = TSNE(n_components=2, random_state=0)
	np.set_printoptions(suppress=True)
	 
	Y=model.fit_transform(X)
	 
	 
	plt.scatter(Y[:, 0], Y[:, 1], c=labels, s=290,alpha=.5)
	 
	plt.show()