from sklearn.cluster import MeanShift
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

def plotMeanShift(MSValues, MSClasses):

	X = MSValues
	y = MSClasses.values.flatten()

	#Play with the parameters here
	#default settings: eps=0.5, min_samples=5
	MSModel = MeanShift().fit(X)
	
	MSexpected = y
	MSpredicted = MSModel.labels_
	
	#print(MSexpected)
	#print(MSpredicted)

	f_performance_measures = open("Clustering/MS_performance_measures.txt", 'w')
	f_performance_measures.write("-----------------Model----------------")
	f_performance_measures.write("\n")
	f_performance_measures.write(MSModel.__str__())
	f_performance_measures.write("\n")
	f_performance_measures.write("--------------------------------------")
	f_performance_measures.write("\n")
	f_performance_measures.write("Homogeneity: %0.3f" % metrics.homogeneity_score(MSexpected, MSpredicted))
	f_performance_measures.write("\n")
	f_performance_measures.write("Completeness: %0.3f" % metrics.completeness_score(MSexpected, MSpredicted))
	f_performance_measures.write("\n")
	f_performance_measures.write("V-measure: %0.3f" % metrics.v_measure_score(MSexpected, MSpredicted))
	f_performance_measures.write("\n")
	f_performance_measures.write("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(MSexpected, MSpredicted))
	f_performance_measures.write("\n")
	f_performance_measures.write("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(MSexpected, MSpredicted))
	f_performance_measures.write("\n")
	f_performance_measures.write("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, MSpredicted))
	f_performance_measures.close()
	
	unique_labels = set(y)
	colors = [plt.cm.Spectral(each)
		for each in np.linspace(0, 1, len(unique_labels))]
	for k, col in zip(unique_labels, colors):
		class_member_mask = (y == k)
		
		xy = X[class_member_mask]
		plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
				markeredgecolor='k', markersize=6)
	
	plt.savefig("Clustering/cluster_MeanShift.png")
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
plotMeanShift(values, classes)