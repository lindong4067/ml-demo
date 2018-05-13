from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

# D:\PycharmProjects\ml-demo\DecisionTre\demo.csv

allElectronicsData = open('D:\PycharmProjects\ml-demo\DecisionTree\demo.csv', 'r')
reader = csv.reader(allElectronicsData, delimiter='\t')
# lists = list(readers)
# print(lists)
headers = next(reader)
print(headers)

featureList = []
labelList = []

for row in reader:
    # print(row[len(row) - 1])
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1, len(row) - 1):
        # print(row[i])
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

print(featureList)

vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
print(str(dummyX))

lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print(str(dummyY))

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyY, dummyX)
print(str(clf))

# with open('allElectronicInformationGainOri.dot', 'w') as f:
#     f = tree.export_graphviz(clf, feature_names=featureList, out_file=f)

oneRowX = dummyX[0]
print(oneRowX)

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print(str(newRowX))


predictedY = clf.predict(newRowX)
print(str(predictedY))


