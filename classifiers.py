from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import  numpy as np 
from sklearn import datasets
# iris model

iris = datasets.load_iris()

X = iris.data
Y = iris.target

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.5)
# Different classifiers
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_perceptron = Perceptron()
clf_KNN = KNeighborsClassifier()

# Training the model
clf_tree.fit(X_train,Y_train)
clf_svm.fit(X_train,Y_train)
clf_perceptron.fit(X_train,Y_train)
clf_KNN.fit(X_train,Y_train)

# Predicting

# Pred tree
pred_tree = clf_tree.predict(X_test)
acc_tree = accuracy_score(Y_test, pred_tree) * 100

print ('Accuracy for tree: {}' .format(acc_tree))

# Pred svm
pred_svm = clf_svm.predict(X_test)
acc_svm = accuracy_score(Y_test, pred_svm) * 100

print ('Accuracy for svm: {}' .format(acc_svm))

# Pred perceptron
pred_perceptron = clf_perceptron.predict(X_test)
acc_perceptron = accuracy_score(Y_test, pred_perceptron) * 100

print ('Accuracy for perceptron: {}' .format(acc_perceptron))

# Pred KNN
pred_KNN = clf_KNN.predict(X_test)
acc_KNN = accuracy_score(Y_test, pred_KNN) * 100

print ('Accuracy for KNN: {}' .format(acc_KNN))

# Best classifier for iris

index = np.argmax([acc_svm,acc_perceptron,acc_KNN,acc_tree])
classifiers = {0:'SVM', 1: 'Perceptron', 2: 'KNN', 3: 'Tree'}
print ('Best iris classifier is {}'.format(classifiers[index]))