from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC

iris = datasets.load_iris()

X, y = iris.data, iris.target

print "X is",len(X)
print "y is",len(y)

# OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)
# OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)
#
# clf = OutputCodeClassifier(LinearSVC(random_state=0),code_size=2, random_state=0)
# clf.fit(X, y).predict(X)

# Some estimators also support multioutput-multiclass classification tasks Decision Trees, Random Forests, Nearest Neighbors.