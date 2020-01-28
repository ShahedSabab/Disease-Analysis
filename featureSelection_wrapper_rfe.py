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
from sklearn.feature_selection import RFECV
from sklearn import preprocessing
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def performanceEvaluation(model, X, y, cv_, n):
    df_temp=pd.DataFrame(columns=['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'nFeatures'])
    model_name = str(model)
    model_name = model_name.split('(')
    model_name = model_name[0]
    print("Model __: ", model_name)
    accuracy = model_selection.cross_val_score(model, X, y, cv=cv_, scoring='accuracy')
    auc = model_selection.cross_val_score(model, X, y, cv=cv_, scoring='roc_auc')
    precision =  model_selection.cross_val_score(model, X, y, cv=cv_, scoring='precision_macro')
    recall =  model_selection.cross_val_score(model, X, y, cv=cv_, scoring='recall_macro')
    f1 =  model_selection.cross_val_score(model, X, y, cv=cv_, scoring='f1_macro')
    
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
    
    
    d =[ model_name, acc ,  a,  p, r,  f, n]
    df_temp = df_temp.append(pd.Series(d,index=['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1','nFeatures']),ignore_index=True)
    return df_temp

    

path = r'''R:\temp\Disease Analysis\parkinson.csv'''
dataset = pd.read_csv(path)
r = Path("mydir/")
absolute = r.absolute()

print(dataset.describe())
#print(dataset.hist())
df=pd.DataFrame(columns=['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'nFeatures'])

df_rfe=pd.DataFrame(columns=['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'nFeatures'])


X = dataset[dataset.columns[:-1]].values
y = dataset[dataset.columns[-1]].values
cv = 10
#X = preprocessing.scale(X)


# Before feature selection
#Logistic Regression
model = LogisticRegression(random_state=0)
df = df.append(performanceEvaluation(model, X, y, cv, len(X[0][:])))

#Random Forest
model = RandomForestClassifier(n_estimators = 100, random_state=0)
df = df.append(performanceEvaluation(model, X, y, cv, len(X[0][:])))


## Feature selection: Recursive Feature Elimination
#Logistic Regression
model = LogisticRegression(random_state=0)
rfecv = RFECV(estimator=model, step=1, cv=cv,scoring='accuracy')
rfecv.fit(X, y)
X_new = rfecv.transform(X)
df_rfe = df_rfe.append(performanceEvaluation(model, X_new, y, cv, len(X_new[0][:])))

# =============================================================================
#Comment out to get more insights

# #opt = rfecv.n_features_
# #num = rfecv.support_
# #sc = rfecv.grid_scores_
# #est = rfecv.estimator_
# =============================================================================
 

#Random Forest
model = RandomForestClassifier(n_estimators = 100, random_state=0)
rfecv = RFECV(estimator=model, step=1, cv=cv,scoring='accuracy')
rfecv.fit(X, y)
X_new = rfecv.transform(X)
df_rfe = df_rfe.append(performanceEvaluation(model, X_new, y, cv, len(X_new[0][:])))

df_rfe.to_csv (r'R:\temp\Disease Analysis\wrapper.csv', index = None, header=True)


