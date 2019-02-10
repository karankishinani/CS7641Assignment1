import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the Dataset
from sklearn import metrics, preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

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
for label in ['income']:
    dataset[label] = preprocessing.LabelEncoder().fit(dataset[label]).transform(dataset[label])

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

decision_tree = DecisionTreeClassifier(max_depth= 10)
decision_tree.fit(X_train, y_train)

neural_network = MLPClassifier(activation = 'logistic', solver='adam',
                    alpha=1e-4, hidden_layer_sizes=(5, 2),
                    learning_rate  = 'invscaling',
                    random_state=1, warm_start = True)
neural_network.fit(X_train, y_train)

adaboost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
                         algorithm="SAMME", n_estimators=100, learning_rate=1)
adaboost.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)

svm_rbf = SVC(kernel = 'rbf', C = 1, tol = 1e-3, probability=True)
print y_train
svm_rbf.fit(X_train, y_train)
print "no issue"

svm_linear = SVC(kernel = "linear", probability=True)
svm_linear.fit(X_train, y_train)

# overall_eval = pd.concat([decision_tree, neural_network, adaboost, knn, svm_rbf, svm_linear], axis = 0)
# overall_eval.sort_values(by = ['f_measure', 'accuracy'], ascending = False, inplace = True)
# print "Comparison:"
# print overall_eval


def generateRoc(test_data, test_label, classifiers, pred_labels, plot_labels, limiter):
    color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan']

    plt.figure()

    for i in range(len(classifiers)):

        if plot_labels[i] not in limiter:
            continue

        y_score = classifiers[i].predict_proba(test_data)[:, 1]
        #pos_class_index = list(np.unique(pred_labels[i])).index(1)

        fpr, tpr, thres = metrics.roc_curve(test_label, y_score, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        lw = 2
        plt.plot(fpr, tpr, color=color[i % len(color)], lw=lw, label=plot_labels[i] + ' AUC = %0.3f' % roc_auc )
        print plot_labels[i] + ' AUC = %0.3f' % roc_auc

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Figure 7: Receiver operating characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig("figure8a1.png")
    plt.xlim(0, 0.4)
    plt.ylim(0.6, 1)
    plt.title('Figure 8: Receiver operating characteristic (ROC) zoomed in at top left')
    plt.savefig("figure8b1.png")

classifier_list = [decision_tree, neural_network, adaboost, knn, svm_rbf, svm_linear]
pred_list = [decision_tree.predict(X_test), neural_network.predict(X_test), adaboost.predict(X_test), knn.predict(X_test), svm_rbf.predict(X_test), svm_linear.predict(X_test)]
clf_labels = ["Decision Tree", "Neural Network", "AdaBoost", "kNN", "SVM - rbf", "SVM - linear"]
limiter = ["Decision Tree", "Neural Network", "AdaBoost", "kNN", "SVM - rbf", "SVM - linear"]
generateRoc(X_test, y_test, classifier_list, pred_list, clf_labels, limiter)