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

# train_scores = []
# test_scores = []
#
# for n in range(2, 40, 2):
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
# plt.title("Figure 5: KNN Accuracy vs k-value")
# plt.xlabel("k-value")
# plt.ylabel("Score")
# plt.grid()
# plt.plot(range(2, 40, 2), train_scores, 'o-', color="r",
#              label="Training score")
# plt.plot(range(2, 40, 2), test_scores, 'o-', color="g",
#              label="Test score")
# plt.legend(loc="best")
# plt.savefig('figure5a.png')
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
plt.savefig('figure5a.png')

plt.clf()  # clear figure

k = 30

classifier = KNeighborsClassifier(n_neighbors=k)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print "Accuracy   :", accuracy_score(y_test, y_pred) * 100
print "Train Score: ", classifier.score(X_train, y_train)
print "CV Score   : ", cross_val_score(classifier, X_train, y_train, cv=3)
print "Test  Score: ", classifier.score(X_test, y_test)



title = "Figure 5b: KNN Learning Curve with k = " + str(k)
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = classifier
plotlearningcurve.plot_learning_curve(estimator, title, X_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
plt.savefig('figure5b.png')

print "== END =="