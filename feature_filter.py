import pandas as pd

# Get classes from given values.
def filterClasses(values):
	filtered = values.drop(values.columns[len(values.columns)-1], axis=1)
	filteredClass = pd.DataFrame(values.iloc[:,-1])
	return filtered, filteredClass

# Filter out attributes that are given in a list.
def filterAttributes(values, attributes, names):
	filterIndices = []
	for a in attributes:
		if a in names:
			filterIndices.append(names.index(a))
	return values.drop(filterIndices, axis=1)
