import numpy as np
from sklearn import tree
from collections import defaultdict
import pydotplus

def extractDataSet(filename):
	X = []
	Y = []
	# parse input

	attributeValues = defaultdict(dict)
	with open(filename, "r", encoding='utf-8') as f:
		for line in f:
			splitLine = line.split()
			curX = []
			for attribute in range(len(splitLine)-1):
				value = splitLine[attribute]
				if value not in attributeValues[attribute]:
					attributeValues[attribute][value] = len(attributeValues[attribute])

				curX.append(attributeValues[attribute][value])
			X.append(curX)
			Y.append(splitLine.pop())
	return X, Y, attributeValues

def generatePDF(clf, filename):
	dot_data = tree.export_graphviz(clf, out_file=None)
	graph = pydotplus.graph_from_dot_data(dot_data)
	graph.write_pdf(filename)

def main():
	X, Y, attributeValues = extractDataSet("dataSet.txt")

	# create decision tree
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X, Y)

	# generate a pdf file for the tree
	generatePDF(clf, "tree.pdf")

	mySample = ["No", "No", "No", "Yes",
				"Some", "$$$", "No", "Yes",
				"Burger", "10–30"]

	# format sample data so that it can be used in clf.predict
	# since it has to be numeric values
	for attribute in range(len(mySample)):
		value = mySample[attribute]
		mySample[attribute] = attributeValues[attribute][value]
	print(clf.predict(np.array(mySample).reshape(1,-1)))

if __name__ == "__main__":
	main()