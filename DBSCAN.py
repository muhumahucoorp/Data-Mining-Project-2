from sklearn.cluster import DBSCAN
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

def plotDBSCAN(DBSValues, DBSClasses):

	X = DBSValues
	y = DBSClasses

	#Play with the parameters here
	DBSModel = DBSCAN().fit(X)
	#DBSModel = DBSCAN(eps=0.3, min_samples=10)

	DBSexpected = y.values.flatten()
	DBSpredicted = DBSModel.labels_
	
	print(DBSexpected)
	print(DBSpredicted)

	f_performance_measures = open("Clustering/DBS_performance_measures.txt", 'w')
	f_performance_measures.write("-----------------Model----------------")
	f_performance_measures.write("\n")
	f_performance_measures.write(DBSModel.__str__())
	f_performance_measures.write("\n")
	f_performance_measures.write("--------------------------------------")
	f_performance_measures.write("\n")
	f_performance_measures.write("Homogeneity: %0.3f" % metrics.homogeneity_score(DBSexpected, DBSpredicted))
	f_performance_measures.write("\n")
	f_performance_measures.write("Completeness: %0.3f" % metrics.completeness_score(DBSexpected, DBSpredicted))
	f_performance_measures.write("\n")
	f_performance_measures.write("V-measure: %0.3f" % metrics.v_measure_score(DBSexpected, DBSpredicted))
	f_performance_measures.write("\n")
	f_performance_measures.write("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(DBSexpected, DBSpredicted))
	f_performance_measures.write("\n")
	f_performance_measures.write("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(DBSexpected, DBSpredicted))
	f_performance_measures.write("\n")
	f_performance_measures.write("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, DBSpredicted))
	f_performance_measures.close()

names = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "class"]
values = pd.read_csv('glass.data', delimiter=',', header=None)

values = values.sample(frac=1).reset_index(drop=True)
values, classes = ff.filterClasses(values)

#Decomment to filter atrributes
#values = ff.filterAttributes(values, ["fLength", "fWidth", "fConc1", "fConc", "fDist"], names)

#Decomment to test classifier and plot roc curve
plotDBSCAN(values, classes)

#Decomment to plot decision boundaries
#plotDecisionBoundaries(values, classes, names)
