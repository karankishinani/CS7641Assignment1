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

from sklearn.tree import DecisionTreeClassifier


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

classifier = DecisionTreeClassifier(max_depth=8, criterion='entropy')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print "Confusion Matrix:"
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print "Accuracy   :", accuracy_score(y_test, y_pred) * 100
print "Train Score: ", classifier.score(X_train, y_train)
print "CV Score   : ", cross_val_score(classifier, X_train, y_train, cv=3)
print "Test  Score: ", classifier.score(X_test, y_test)


title = "Figure 2b: Decision Tree Learning Curve with max_depth = 8"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = classifier
plotlearningcurve.plot_learning_curve(estimator, title, X_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
plt.savefig('figure2b.png')

# This section is referenced from my Machine Learning for Trading Assignment Fall 2018
import math
train_errors = []
test_errors = []
for depth_limit in range(1, 20 + 1):
    # create a learner and train it
    # learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
    learner = DecisionTreeClassifier(max_depth= depth_limit, criterion='entropy')
    learner.fit(X_train, y_train)
    # print learner.author()

    train_errors.append(1-learner.score(X_train, y_train))
    test_errors.append(1-np.mean(cross_val_score(learner, X_train, y_train, cv=3)))


plt.figure()
plt.clf()
plt.plot(train_errors, label="Training Error")
plt.plot(test_errors, label="Cross Validation Error")
plt.legend(loc=4)
#plt.ylim([0, 0.009])
plt.title("Figure 1b: Decision Tree Error vs Tree Depth")
plt.xlabel("Tree Depth")
plt.ylabel("Error")
plt.savefig('figure1b.png')
plt.clf()  # clear figure

print "== END =="