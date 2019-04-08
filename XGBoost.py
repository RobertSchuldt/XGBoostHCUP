# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 17:06:39 2018

@author: robsc
"""

#Import via chunk size 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import xgboost as xgb
import graphviz 

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from xgboost import plot_tree

start_time = time.time()


dataset = pd.read_sas("playset.sas7bdat", format = 'sas7bdat', encoding = 'utf-8')
    

# Importing the dataset

ds_nd = dataset.loc[:, ['IM_AGE', 'IM_LOS', 'IM_NDX', 'IM_NPR', 'IM_TOTCHG' ,'IM_AWEEKEND', 'IM_ELECTIVE', 'IM_FEMALE']].values
ds_nd = pd.DataFrame(ds_nd)

ds_dd = dataset.loc[:, ['ASOURCE', 'DX1', 'ATYPE',  'HOSPID', 'HOSPST', 'IM_RACE' ,'IM_ZIPINC_QRTL' ,'IM_PL_UR_CAT4']].values
ds_dd = pd.DataFrame(ds_dd)

ds_dd = pd.get_dummies(ds_dd, drop_first = True)

dataset = ds_dd.join(ds_nd, how = 'right')


X = dataset.iloc[:, :]

dataset2 = pd.read_sas("playset.sas7bdat", format = 'sas7bdat', encoding = 'utf-8')

y = dataset2.loc[:, 'DIED']
y.describe()

y = dataset2.loc[:, 'DIED'].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting XGBoost to the Training set

classifier = XGBClassifier(silent = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

print(classifier)

print("XGBoost Classifier Took", time.time() - start_time, "to run")




#Shows feature importance
xgb.plot_importance(classifier)
plt.show()



#shows the tree 
xgb.plot_tree(loaded_classifier, num_trees=2)
plt.show()

""""
#save the model results to recall in future
pickle.dump(classifier, open("nis_hospital.pickle.dat", "wb"))
"""

#reload the model witht he following command
loaded_classifier = pickle.load(open("nis_hospital.pickle.dat", "rb"))