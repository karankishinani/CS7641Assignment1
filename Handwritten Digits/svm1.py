import plotlearningcurve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


dataset_train = pd.read_csv('pendigits.tra', sep= ',', header=None)

dataset_test = pd.read_csv('pendigits.tes', sep= ',', header=None)

print dataset_test.shape, dataset_train.shape

dataset = dataset_train.append(dataset_test)


#print len(dataset)
#
X_train = dataset_train.values[:,:-2]
y_train = dataset_train.values[:,-1]
X_test = dataset_test.values[:,:-2]
y_test = dataset_test.values[:,-1]
#
# X = dataset.values[:,:-2]
# y = dataset.values[:,-1]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

train_scores = []
test_scores = []

classifier = SVC(kernel='rbf', C = 1, tol = 1e-3, gamma=0.001)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print "Confusion Matrix:"
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print "Accuracy   :", accuracy_score(y_test, y_pred) * 100
print "Train Score: ", classifier.score(X_train, y_train)
print "CV Score   : ", cross_val_score(classifier, X_train, y_train, cv=3)
print "Test  Score: ", classifier.score(X_test, y_test)

title = "Figure 6c: Support Vector Machine Learning Curve with rbf kernel"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = classifier
plotlearningcurve.plot_learning_curve(estimator, title, X_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
plt.savefig('figure6c.png')

print "== END =="