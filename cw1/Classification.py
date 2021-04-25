# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 1. Classification
# 1.1
df = pd.read_csv('adult.csv')
df = df.drop(['fnlwgt'],axis=1)
# number of instances
M = len(df)
print('Number of instances: ',M)
# number of missing values
num_missing_value = df.iloc[:,:13].isna().sum().sum()
print('Number of missing values: ',num_missing_value)
# fraction of missing values over all attribute values
frac_missing_value = num_missing_value/(M*(len(df.columns)-1))
print('Fraction of missing values over all attribute values: ',frac_missing_value)
# number of instances with missing values
num_instan = df.iloc[:,:13].T.isna().any().sum()
print('Number of instances with missing values',num_instan)
# fraction of instances with missing values over all instances
frac_instan = num_instan/M
print('Fraction of instances with missing values over all instances: ',frac_instan)

# 1.2
# ignore missing values since they are invalid  
labels = []
for i in range(13):
    values = df.iloc[:,i].dropna().unique()
    le = LabelEncoder().fit(values)
    print('The set of all possible values for {}: {}'.format(df.columns[i],sorted(values)))
    labels.append(le)
    
# 1.3
# ignore any instance with missing values
df2 = df.iloc[~df.T.isna().any().values,:]
# encode 'class' to nominal
for i in range(13):
    df2.iloc[:,i] = labels[i].transform(df2.iloc[:,i])
df2.iloc[:,13] = LabelEncoder().fit_transform(df2.iloc[:,13])
# split dataset into training set and testing set
x_train, x_test, y_train, y_test = train_test_split(df2.iloc[:,:13],df2.iloc[:,13],random_state=0)
# initialize the decision tree
clf = tree.DecisionTreeClassifier(random_state=0)
# fit the tree model to the training data
clf.fit(x_train,y_train)
# compute error rate
print('training error rate = ',1-clf.score(x_train,y_train))
print('test error rate = ',1-clf.score(x_test,y_test))

# 1.4
missing_index = df.T.isna().any().values # indexes of instances with at least one missing value
df_1 = df.fillna('missing')# dataset for construcing tree 1
popular_value = {}
for i in range(13):
    popular_value[df.columns[i]] = df.iloc[:,i].mode()[0]
df_2 = df.fillna(popular_value)# dataset for constructing tree 2
# encode both datasets
for i in range(len(df_1.columns)):
    df_1.iloc[:,i] = LabelEncoder().fit_transform(df_1.iloc[:,i])
for i in range(len(df_2.columns)):
    df_2.iloc[:,i] = LabelEncoder().fit_transform(df_2.iloc[:,i])
# instances with at least one missing value and non-missing value in each dataset
missing_instan_1 = df_1.iloc[missing_index]
non_missing_instan_1 = df_1.iloc[~missing_index]
missing_instan_2 = df_2.iloc[missing_index]
non_missing_instan_2 = df_2.iloc[~missing_index]
# construct train set and test set for tree 1
# construct the test set by taking a sample subset from those in the original dataset
# that are not included in the training set
D1_train = pd.concat([missing_instan_1,non_missing_instan_1.sample(len(missing_instan_1))])
D1_test = df_1[~df_1.index.isin(D1_train.index)].sample(round(len(D1_train)/3))
# train set and test set for tree 2
# In order to better compare the two trees' performances, use the same instances as test sets for both trees
D2_train = pd.concat([missing_instan_2,non_missing_instan_2.sample(len(missing_instan_2))])
D2_test = df_2[df_2.index.isin(D1_test.index)]
# train decision tree 1
clf1 = tree.DecisionTreeClassifier(random_state=0)
clf1.fit(D1_train.iloc[:,:13],D1_train.iloc[:,13])
# train decision tree 2
clf2 = tree.DecisionTreeClassifier(random_state=0)
clf2.fit(D2_train.iloc[:,:13],D2_train.iloc[:,13])
# compute error rates of each tree
print('test error rate of tree1 = ',1-clf1.score(D1_test.iloc[:,:13],D1_test.iloc[:,13]))
print('test error rate of tree2 = ',1-clf2.score(D2_test.iloc[:,:13],D2_test.iloc[:,13]))