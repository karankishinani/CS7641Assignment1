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

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# Importing the Dataset

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

dataset_train = pd.read_csv('adult.data', sep= ',', names= names)

dataset_test = pd.read_csv('adult.test', sep= ',', names= names, skiprows=[0])

dataset_test['income'] = dataset_test['income'].str[:-1]

dataset = pd.concat([dataset_train, dataset_test])

# Setting all the categorical columns to type category
for col in set(dataset.columns) - set(dataset.describe().columns):
    dataset[col] = dataset[col].astype('category')

# Dropping the Missing Values
dataset['native-country'] = dataset['native-country'].replace(' ?',np.nan)
dataset['workclass'] = dataset['workclass'].replace(' ?',np.nan)
dataset['occupation'] = dataset['occupation'].replace(' ?',np.nan)


dataset.dropna(how='any',inplace=True)


# Remove education and fnlwgt
dataset.drop(labels = ['education','fnlwgt','hours-per-week'], axis = 1, inplace = True)

#print(dataset.shape)
#print(dataset.head())

# from sklearn import preprocessing
# labels = [
#     'workclass',
#     'education',
#     'marital-status',
#     'occupation',
#     'relationship',
#     'race',
#     'sex',
#     'native-country',
#     'income',
# ]
#
# for label in labels:
#     dataset[label] = preprocessing.LabelEncoder().fit(dataset[label]).transform(dataset[label])

# print(dataset.shape)
# print(dataset.head())

X = dataset.drop('income', axis=1)
y = np.asarray(dataset['income'])

# Representing multi-class categorical features as binary features

X_cat_1hot = pd.get_dummies(X.select_dtypes(['category']))
X_non_cat = X.select_dtypes(exclude = ['category'])

X = pd.concat([X_non_cat, X_cat_1hot], axis=1, join='inner')

# Split into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Normalizing the dataset
scaler = StandardScaler()

# Fitting on the training data
scaler.fit(X_train, y_train)

# Applying transformation to the training and test data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
                         algorithm="SAMME", n_estimators=100, learning_rate=1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print "Confusion Matrix:"
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print "Accuracy   :", accuracy_score(y_test, y_pred) * 100
print "Train Score: ", classifier.score(X_train, y_train)
print "CV Score   : ", cross_val_score(classifier, X_train, y_train, cv=3)
print "Test  Score: ", classifier.score(X_test, y_test)


title = "Figure 4: AdaBoost Learning Curve"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = classifier
plotlearningcurve.plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
plt.savefig('figure4a.png')


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