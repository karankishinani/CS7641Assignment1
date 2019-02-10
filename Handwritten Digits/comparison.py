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

decision_tree = DecisionTreeClassifier(max_depth=8, criterion='entropy')
decision_tree.fit(X_train, y_train)

neural_network = MLPClassifier(hidden_layer_sizes=(100, 50, 100), max_iter=1000)
neural_network.fit(X_train, y_train)

adaboost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
                         algorithm="SAMME", n_estimators=100, learning_rate=1)
adaboost.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

svm_rbf = SVC(kernel='rbf', C = 1, tol = 1e-3, gamma=0.001, probability=True)
print y_train
svm_rbf.fit(X_train, y_train)
print "no issue"

svm_linear = SVC(kernel = "linear", C = 1, tol = 1e-3, gamma=0.001, probability=True)
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
        plt.plot(fpr, tpr, color=color[i % len(color)], lw=lw, label=plot_labels[i] + ' AUC = %0.2f' % roc_auc )
        print plot_labels[i] + ' AUC = %0.2f' % roc_auc

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Figure 9: Receiver operating characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig("figure9.png")
    plt.xlim(0, 0.2)
    plt.ylim(0.9, 1)
    plt.title('Figure 10: Receiver operating characteristic (ROC) zoomed in at top left')
    plt.savefig("figure10.png")

classifier_list = [decision_tree, neural_network, adaboost, knn, svm_rbf, svm_linear]
pred_list = [decision_tree.predict(X_test), neural_network.predict(X_test), adaboost.predict(X_test), knn.predict(X_test), svm_rbf.predict(X_test), svm_linear.predict(X_test)]
clf_labels = ["Decision Tree", "Neural Network", "AdaBoost", "kNN", "SVM - rbf", "SVM - linear"]
limiter = ["Decision Tree", "Neural Network", "AdaBoost", "kNN", "SVM - rbf", "SVM - linear"]
generateRoc(X_test, y_test, classifier_list, pred_list, clf_labels, limiter)