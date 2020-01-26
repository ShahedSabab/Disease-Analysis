# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 02:24:23 2020

@author: sabab
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 02:58:53 2020

@author: sabab
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from statistics import mean 
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler


def performanceEvaluation(model, X, y, cv):
    df_temp=pd.DataFrame(columns=['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1'])
    model_name = str(model)
    model_name = model_name.split('(')
    model_name = model_name[0]
    print("Model __: ", model_name)
    accuracy = model_selection.cross_val_score(model, X, y, cv=10, scoring='accuracy')
    auc = model_selection.cross_val_score(model, X, y, cv=10, scoring='roc_auc')
    precision =  model_selection.cross_val_score(model, X, y, cv=10, scoring='precision_macro')
    recall =  model_selection.cross_val_score(model, X, y, cv=10, scoring='recall_macro')
    f1 =  model_selection.cross_val_score(model, X, y, cv=10, scoring='f1_macro')
    
# =============================================================================
#     acc = "{:.3f} ({:.3f})".format(accuracy.mean(), accuracy.std())
#     a = "{:.3f} ({:.3f})".format(auc.mean(), auc.std())
#     p = "{:.3f} ({:.3f})".format(precision.mean(), precision.std())
#     r = "{:.3f} ({:.3f})".format(recall.mean(), recall.std())
#     f = "{:.3f} ({:.3f})".format(f1.mean(), f1.std())
# =============================================================================
    
    acc = accuracy.mean()
    a = auc.mean()
    p = precision.mean()
    r = recall.mean()
    f = f1.mean()
    
    
    d =[ model_name, acc ,  a,  p, r,  f]
    df_temp = df_temp.append(pd.Series(d,index=['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1']),ignore_index=True)
    return df_temp


path = r'''T:\U of M\parkinson.csv'''
dataset = pd.read_csv(path)

print(dataset.describe())
#print(dataset.hist())
df=pd.DataFrame(columns=['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1'])

df_chi2=pd.DataFrame(columns=['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1'])

df_f=pd.DataFrame(columns=['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1'])

df_mi=pd.DataFrame(columns=['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1'])

X = dataset[dataset.columns[:-1]].values
y = dataset[dataset.columns[-1]].values
cv = 10
feature_n = 10

## Before feature selection
#Logistic Regression
model = LogisticRegression(random_state=0)
df = df.append(performanceEvaluation(model, X, y, cv))

#Random Forest
model = RandomForestClassifier(n_estimators = 100, random_state=0)
df = df.append(performanceEvaluation(model, X, y, cv))

#SVM
model = SVC(gamma='auto', random_state=0)
df = df.append(performanceEvaluation(model, X, y, cv))

#Naive Bayes
model = GaussianNB()
df = df.append(performanceEvaluation(model, X, y, cv))

#KNN
model = KNeighborsClassifier(n_neighbors=5)
df = df.append(performanceEvaluation(model, X, y, cv))



## feature selection: Filter method-chi2
#normalized feature within a fange from 0 to 1
X_new = MinMaxScaler().fit_transform(X)

X_new = SelectKBest(chi2, k=feature_n).fit_transform(X_new,y)

#Logistic Regression
model = LogisticRegression(random_state=0)
df_chi2 = df_chi2.append(performanceEvaluation(model, X_new, y, cv))

#Random Forest
model = RandomForestClassifier(n_estimators = 100, random_state=0)
df_chi2 = df_chi2.append(performanceEvaluation(model, X_new, y, cv))

#SVM
model = SVC(gamma='auto', random_state=0)
df_chi2 = df_chi2.append(performanceEvaluation(model, X_new, y, cv))

#Naive Bayes
model = GaussianNB()
df_chi2 = df_chi2.append(performanceEvaluation(model, X_new, y, cv))

#KNN
model = KNeighborsClassifier(n_neighbors=5)
df_chi2 = df_chi2.append(performanceEvaluation(model, X_new, y, cv))


## feature selection: Filter method-ANOVA ftest

X_new = SelectKBest(f_classif, k=feature_n).fit_transform(X,y)


#Logistic Regression
model = LogisticRegression(random_state=0)
df_f = df_f.append(performanceEvaluation(model, X_new, y, cv))

#Random Forest
model = RandomForestClassifier(n_estimators = 100, random_state=0)
df_f = df_f.append(performanceEvaluation(model, X_new, y, cv))

#SVM
model = SVC(gamma='auto', random_state=0)
df_f = df_f.append(performanceEvaluation(model, X_new, y, cv))

#Naive Bayes
model = GaussianNB()
df_f = df_f.append(performanceEvaluation(model, X_new, y, cv))

#KNN
model = KNeighborsClassifier(n_neighbors=5)
df_f = df_f.append(performanceEvaluation(model, X_new, y, cv))


## feature selection: Filter method-Mutual Information 

X_new = SelectKBest(mutual_info_classif, k=feature_n).fit_transform(X,y)


#Logistic Regression
model = LogisticRegression(random_state=0)
df_mi = df_mi.append(performanceEvaluation(model, X_new, y, cv))

#Random Forest
model = RandomForestClassifier(n_estimators = 100, random_state=0)
df_mi = df_mi.append(performanceEvaluation(model, X_new, y, cv))

#SVM
model = SVC(gamma='auto', random_state=0)
df_mi = df_mi.append(performanceEvaluation(model, X_new, y, cv))

#Naive Bayes
model = GaussianNB()
df_mi = df_mi.append(performanceEvaluation(model, X_new, y, cv))

#KNN
model = KNeighborsClassifier(n_neighbors=5)
df_mi = df_mi.append(performanceEvaluation(model, X_new, y, cv))