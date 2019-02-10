import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as preprocessing

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
print("How many of each class do we have:"+str(np.unique(dataset_train['income'],return_counts=True)))
dataset_test = pd.read_csv('adult.test', sep= ',', names= names, skiprows=[0])
print("How many of each class do we have:"+str(np.unique(dataset_test['income'],return_counts=True)))
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

# Calculate the correlation and plot it
sns.heatmap(dataset.corr(), square=True, center=1)
plt.title('Figure 0: Correlation in Adult Dataset')
plt.savefig('figure0.png', bbox_inches='tight')


# Remove education and fnlwgt
dataset.drop(labels = ['education','fnlwgt','hours-per-week'], axis = 1, inplace = True)


X = dataset.drop('income', axis=1)
y = np.asarray(dataset['income'])

print("Amount of classes:"+str(np.unique(y,return_counts=True)))
