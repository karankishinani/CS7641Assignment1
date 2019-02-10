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

names = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income',
]

dataset = pd.read_csv(
'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                           sep= ',', names= names)

dataset_test = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
                           sep= ',', names= names, skiprows=[0])
dataset_test['income'] = dataset_test['income'].str[:-1]

dataset = dataset.append(dataset_test)

dataset['country'] = dataset['country'].replace(' ?',np.nan)
dataset['workclass'] = dataset['workclass'].replace(' ?',np.nan)
dataset['occupation'] = dataset['occupation'].replace(' ?',np.nan)

dataset.dropna(how='any',inplace=True)

print(dataset.shape)
#print(dataset.head())

from sklearn import preprocessing
labels = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income',
]

for label in labels:
    dataset[label] = preprocessing.LabelEncoder().fit(dataset[label]).transform(dataset[label])

# print(dataset.shape)
# print(dataset.head())

X = dataset.drop('income', axis=1)
y = dataset['income']

# X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='poly',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)


classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print "Accuracy   :", accuracy_score(y_test, y_pred) * 100
print "Train Score: ", classifier.score(X_train, y_train)
print "CV Score   : ", cross_val_score(classifier, X_train, y_train, cv=3)
print "Test  Score: ", classifier.score(X_test, y_test)

# This excerpt is extracted from Scikitlearn webpage

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt




title = "Figure 13: SVM Learning Curve"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)

estimator = classifier
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
plt.savefig('figure14.png')


# # This section is referenced from my Machine Learning for Trading Assignment Fall 2018
# import math
# train_errors = []
# test_errors = []
# for depth_limit in range(1, 50 + 1):
#     # create a learner and train it
#     # learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
#     learner = DecisionTreeClassifier(max_depth= depth_limit)
#     learner.fit(X_train, y_train)
#     # print learner.author()
#
#     train_errors.append(1-learner.score(X_train, y_train))
#     test_errors.append(1-max(cross_val_score(learner, X_train, y_train, cv=3)))
#
#
# plt.figure()
# plt.clf()
# plt.plot(train_errors, label="Training Error")
# plt.plot(test_errors, label="Cross Validation Error")
# plt.legend(loc=4)
# #plt.ylim([0, 0.009])
# plt.title("Figure 1: Decision Tree Error vs Tree Depth")
# plt.xlabel("Tree Depth")
# plt.ylabel("Error")
# plt.savefig('figure1.png')
# plt.clf()  # clear figure

print "== END =="