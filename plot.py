import pandas
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys

print("Prepare data ...")
names = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
values = pandas.read_csv('glass.data', delimiter=',', header=None)
classes = ('building_windows_float_processed', 'building_windows_non_float_processed', 'vehicle_windows_float_processed', 'vehicle_windows_non_float_processed', 'containers', 'tableware', 'headlamps')

classifications = values.iloc[:,-1]
values = values.drop(labels=10, axis=1)
values = values.drop(labels=0, axis=1)

#print(classifications)
#print(values)

maxima = values.max(axis=0)
minima = values.min(axis=0)

f = open('Plots/valueRange.txt', 'w')

print("Start plotting ...")
for i in range(1,len(values.columns)+1):
	# Create the class distribution plot.
	# The thenth coloumn represents the nominal class of the tuples.
	
	# Writes the maxima and minima for each attribute
	f.write('Attribute: ' + names[i] + ', max: ' + str(maxima[i]) + ", min: " + str(minima[i]) + "\n")
	
	"""
	# Create Histograms
	plt.clf()
	plt.xlabel('distribution')
	plt.ylabel(names[i])
	plt.hist(values[i], 100, facecolor='green', alpha=0.95)
	plt.savefig("Plots/" + names[i] + "_histogram")
	plt.clf()
	"""
	
	"""
	# Create Boxplots
	plt.figure()
	plt.boxplot(values[i], 1)
	plt.xticks([1], [names[i]])
	plt.savefig("Plots/" + names[i] + "_boxplot")
	plt.clf()
	"""

f.close()

print("Created plots.")

"""
# Create a heatmap that represents the correlation between the values
print("Heatmap creation ...")
plt.clf()
x = values.corr()
x.columns = names[1:]
x.index = names[1:]
sns.set(font_scale=0.7)
sns.heatmap(x, annot=True, linewidths=0.5)
plt.savefig("Plots/" + "correlation_heatmap")
print("Heatmap created.")
"""
