import plotlearningcurve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


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

param_range = range(1, 40+1, 1)
train_scores, test_scores = validation_curve(
    KNeighborsClassifier(), X_train, y_train, param_name="n_neighbors", param_range=param_range,
    cv=10, scoring="accuracy", n_jobs=4)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.title("Figure 5b: KNN Accuracy vs k-value")
plt.xlabel("k-value")
plt.ylabel("Score")
plt.grid()

lw = 2
plt.plot(param_range, train_scores_mean, 'o-', label="Training score",
             color="r", lw=lw)
plt.plot(param_range, test_scores_mean, 'o-', label="Cross-validation score",
             color="g", lw=lw)
plt.legend(loc="best")
plt.savefig('figure5b.png')

# train_scores = []
# test_scores = []
#
# for n in range(1, 10):
#     knn = KNeighborsClassifier(n_neighbors=n)
#     knn.fit(X_train, y_train)
#     train_score = knn.score(X_train, y_train)
#     test_score = knn.score(X_test, y_test)
#     train_scores.append(train_score)
#     test_scores.append(test_score)
#     print('KNN : Training score - {'+str(train_score)+'} -- Test score - {'+str(test_score)+'}')
#
# plt.figure()
# plt.clf()
# plt.title("Figure 11: KNN Accuracy vs k-value")
# plt.xlabel("k-value")
# plt.ylabel("Score")
# plt.grid()
# plt.plot(range(1, 10), train_scores, 'o-', color="r",
#              label="Training score")
# plt.plot(range(1, 10), test_scores, 'o-', color="g",
#              label="Test score")
# plt.legend(loc="best")
# plt.savefig('figure11.png')
# plt.clf()  # clear figure


k = 3

classifier = KNeighborsClassifier(n_neighbors=k)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print "Confusion Matrix:"
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print "Accuracy   :", accuracy_score(y_test, y_pred) * 100
print "Train Score: ", classifier.score(X_train, y_train)
print "CV Score   : ", cross_val_score(classifier, X_train, y_train, cv=3)
print "Test  Score: ", classifier.score(X_test, y_test)


title = "Figure 5c: kNN Learning Curve with k = 3"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = classifier
plotlearningcurve.plot_learning_curve(estimator, title, X_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
plt.savefig('figure5c.png')

print "== END =="