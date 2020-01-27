# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 02:17:10 2020

@author: sabab
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 01:06:06 2020

@author: sabab
"""

import pandas as pd  
import numpy as np  
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

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

df=pd.DataFrame(columns=['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'nFeatures'])
df_ef=pd.DataFrame(columns=['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'nFeatures'])



X = dataset[dataset.columns[:-1]].values
y = dataset[dataset.columns[-1]].values
cv = 10
feature_n_min = 5
feature_n_max = 20


#Exhaustive Feature Seelector
#Logistic Regression
model = LogisticRegression(random_state=0)
efs = ExhaustiveFeatureSelector(model,min_features=feature_n_min,max_features=feature_n_max,scoring='accuracy',print_progress=True,cv=cv)
efs = efs.fit(X,y)
ff = list(efs.k_feature_idx_)
X_new = dataset[dataset.columns[ff]].values
df_ef = df_ef.append(performanceEvaluation(model, X_new, y, cv, len(X_new[0][:])))


#Random Forest
model = RandomForestClassifier(n_estimators = 100, random_state=0)
efs = ExhaustiveFeatureSelector(model,min_features=feature_n_min,max_features=feature_n_max,scoring='accuracy',print_progress=True,cv=cv)
efs = efs.fit(X,y)
ff = list(efs.k_feature_idx_)
X_new = dataset[dataset.columns[ff]].values
df_ef = df_ef.append(performanceEvaluation(model, X_new, y, cv, len(X_new[0][:])))

df_ef.to_csv (r'R:\temp\Disease Analysis\wrapper_efs.csv', index = None, header=True)
