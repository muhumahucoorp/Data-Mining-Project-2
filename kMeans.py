from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from scipy import interp
from itertools import cycle
import pandas as pd
import numpy as np
import itertools

import feature_filter as ff

def plotkMeans(kMValues, kMClasses):

	X = kMValues
	y = kMClasses

	#Play with the parameters here
	kMModel = KMeans(n_clusters=7).fit(X)

	kMexpected = y.values.flatten()
	kMpredicted = kMModel.labels_
	
	#print(kMexpected)
	#print(kMpredicted)

	f_performance_measures = open("Clustering/kM_performance_measures.txt", 'w')
	f_performance_measures.write("-----------------Model----------------")
	f_performance_measures.write("\n")
	f_performance_measures.write(kMModel.__str__())
	f_performance_measures.write("\n")
	f_performance_measures.write("--------------------------------------")
	f_performance_measures.write("\n")
	f_performance_measures.write("Homogeneity: %0.3f" % metrics.homogeneity_score(kMexpected, kMpredicted))
	f_performance_measures.write("\n")
	f_performance_measures.write("Completeness: %0.3f" % metrics.completeness_score(kMexpected, kMpredicted))
	f_performance_measures.write("\n")
	f_performance_measures.write("V-measure: %0.3f" % metrics.v_measure_score(kMexpected, kMpredicted))
	f_performance_measures.write("\n")
	f_performance_measures.write("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(kMexpected, kMpredicted))
	f_performance_measures.write("\n")
	f_performance_measures.write("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(kMexpected, kMpredicted))
	f_performance_measures.write("\n")
	f_performance_measures.write("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, kMpredicted))
	f_performance_measures.close()

def plotCluster(data):
	kmeans = KMeans(n_clusters=7)
	kmeans.fit(data)
	
	# Step size of the mesh. Decrease to increase the quality of the VQ.
	h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
	
	# Plot the decision boundary. For that, we will assign a color to each
	x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
	y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	
	# Obtain labels for each point in mesh. Use last trained model.
	Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
	
	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.figure(1)
	plt.clf()
	plt.imshow(Z, interpolation='nearest',
			extent=(xx.min(), xx.max(), yy.min(), yy.max()),
			cmap=plt.cm.Paired,
			aspect='auto', origin='lower')
	
	plt.plot(data[:, 0], data[:, 1], 'k.', markersize=2)
	# Plot the centroids as a white X
	centroids = kmeans.cluster_centers_
	plt.scatter(centroids[:, 0], centroids[:, 1],
			marker='x', s=169, linewidths=3,
			color='w', zorder=10)
	plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
			'Centroids are marked with white cross')
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xticks(())
	plt.yticks(())
	plt.show()

names = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "class"]
values = pd.read_csv('glass.data', delimiter=',', header=None)

values = values.sample(frac=1).reset_index(drop=True)
values, classes = ff.filterClasses(values)

#Decomment to filter atrributes
#values = ff.filterAttributes(values, ["fLength", "fWidth", "fConc1", "fConc", "fDist"], names)

#Decomment to test classifier and plot roc curve
plotkMeans(values, classes)

#Decomment to plot clustering
plotCluster(values)
