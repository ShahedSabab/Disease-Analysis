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
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def performanceEvaluation(model, X, y, cv):
    global df_final
    model_name = str(model)
    model_name = model_name.split('(')
    model_name = model_name[0]
    print("Model __: ", model_name)
    accuracy = model_selection.cross_val_score(model, X, y, cv=10, scoring='accuracy')
    auc = model_selection.cross_val_score(model, X, y, cv=10, scoring='roc_auc')
    precision =  model_selection.cross_val_score(model, X, y, cv=10, scoring='precision_macro')
    recall =  model_selection.cross_val_score(model, X, y, cv=10, scoring='recall_macro')
    f1 =  model_selection.cross_val_score(model, X, y, cv=10, scoring='f1_macro')
    
    acc = "{:.3f} ({:.3f})".format(accuracy.mean(), accuracy.std())
    a = "{:.3f} ({:.3f})".format(auc.mean(), auc.std())
    p = "{:.3f} ({:.3f})".format(precision.mean(), precision.std())
    r = "{:.3f} ({:.3f})".format(recall.mean(), recall.std())
    f = "{:.3f} ({:.3f})".format(f1.mean(), f1.std())
    d =[ model_name, acc ,  a,  p, r,  f]
    df_final = df_final.append(pd.Series(d,index=['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1']),ignore_index=True)
    print("Accuracy: {:.3f} ({:.3f})".format(accuracy.mean(), accuracy.std()))
    print("AUC: {:.3f} ({:.3f})".format( auc.mean(), auc.std()))
    print("Precision: {:.3f} ({:.3f})".format( precision.mean(), precision.std()))
    print("Recall: {:.3f} ({:.3f})".format( recall.mean(), recall.std()))
    print("f1: {:.3f} ({:.3f})\n\n\n\n".format( f1.mean(), f1.std()))


path = r'''T:\U of M\parkinson.csv'''
dataset = pd.read_csv(path)
df_final=pd.DataFrame(columns=['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1'])
print(dataset.describe())
#print(dataset.hist())

X = dataset[dataset.columns[:-1]].values
y = dataset[dataset.columns[-1]].values
cv = 10

mae = []
mse = []
rmse = []



#Logistic Regression
model = LogisticRegression(random_state=0)
performanceEvaluation(model, X, y, cv)

#Random Forest
model = RandomForestClassifier(n_estimators = 100, random_state=0)
performanceEvaluation(model, X, y, cv)
#
##SVM
#model = SVC(gamma='auto', random_state=0)
#performanceEvaluation(model, X, y, cv)
#
##Naive Bayes
#model = GaussianNB()
#performanceEvaluation(model, X, y, cv)
#
##KNN
#model = KNeighborsClassifier(n_neighbors=5)
#performanceEvaluation(model, X, y, cv)

print(df_final)
# =============================================================================
# #Stratified split 
# skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
# for train_index, test_index in skf.split(X, y):
#      X_train, X_test = X[train_index], X[test_index]
#      y_train, y_test = y[train_index], y[test_index]
#      lrg.fit(X_train, y_train)
# #     coeff_df = pd.DataFrame(lrg.coef_, X.columns, columns=['Coefficient'])  
#      y_pred = lrg.predict(X_test)
#      df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#      mae.append(metrics.mean_absolute_error(y_test, y_pred))  
#      mse.append(metrics.mean_squared_error(y_test, y_pred))  
#      rmse.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# 
# 
# 
# print(mean(mae))
# print(mean(mse))
# print(mean(rmse))
# 
# plt.figure(figsize=(15,10))
# plt.tight_layout()
# seabornInstance.distplot(dataset['status'])
# dataset['status'].hist()
# =============================================================================
