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
	y = kMClasses.values.flatten()

	#Play with the parameters here
	kMModel = KMeans(n_clusters=len(set(y))).fit(X)

	kMexpected = y
	kMpredicted = kMModel.labels_
	
	buffer = np.matrix(kMpredicted).T
	
	cluster = X.copy()
	cluster = np.concatenate((cluster, buffer), axis = 1)
	
	
	purities = []
	
	for i in np.unique(kMpredicted):
		ith_cluster = np.zeros(shape=(0,11))
		for j in cluster:
			if(j[:,-1] == i):
				ith_cluster = np.concatenate((ith_cluster, j), axis=0)
		mijs = np.sum(np.delete(ith_cluster, 10, axis=1), axis = 0)
		max_mij = np.max(mijs)
		mj = np.sum(mijs, axis=1)
		purities.append(max_mij / mj)
	
	mis = np.sum(X, axis=0)
	m = np.sum(mis)
	
	purity = []
	
	for p in purities:
		purityI = 0;
		for mi in mis:
			purityI += (mi/m) * p
		purity.append(purityI)
	
	f_performance_measures = open("Clustering/kM_performance_measures.txt", 'w')
	f_performance_measures.write("-----------------Model----------------")
	f_performance_measures.write("\n")
	f_performance_measures.write(kMModel.__str__())
	f_performance_measures.write("\n")
	f_performance_measures.write("--------------------------------------")
	f_performance_measures.write("\n")
	f_performance_measures.write("Purities: %0.3f" % purities)
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
	
	
	unique_labels = set(y)
	colors = [plt.cm.Spectral(each)
		for each in np.linspace(0, 1, len(unique_labels))]
	for k, col in zip(unique_labels, colors):
		class_member_mask = (y == k)
		
		xy = X[class_member_mask]
		plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
				markeredgecolor='k', markersize=6)
	
	centroids = kMModel.cluster_centers_
	plt.scatter(centroids[:, 0], centroids[:, 1],
			marker='x', s=169, linewidths=3,
			color='b', zorder=10)
	
	plt.savefig("Clustering/cluster_kMeans.png")
	#plt.show()

names = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "class"]
values = pd.read_csv('glass.data', delimiter=',', header=None)

values = values.sample(frac=1).reset_index(drop=True)
values, classes = ff.filterClasses(values)

#Decomment to filter atrributes
#values = ff.filterAttributes(values, ["fLength", "fWidth", "fConc1", "fConc", "fDist"], names)

normalizer = preprocessing.MinMaxScaler()
values = normalizer.fit_transform(values)

#Decomment to test classifier and plot roc curve
plotkMeans(values, classes)
