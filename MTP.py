# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 20:02:01 2020

@author: Mypc
"""

import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
dataframe = pd.read_excel('Arko shankar.xlsx')

mean_v2 = dataframe["V2"].mean()
mean_v2 = int(mean_v2)
list_per = dataframe.values.tolist()
[j.pop(0) for j in list_per]
for i in range(0,len(list_per)):
    if(list_per[i][0]>mean_v2):
        list_per[i][0]=1
    else:
        list_per[i][0]=0
        
dataframe = DataFrame(list_per)
le =  LabelEncoder()
dataframe.columns
data_x = dataframe.loc[:][:]
data_x = data_x.drop(data_x.columns[-1],axis = 1)
objList = data_x.select_dtypes(include = "object").columns
for feat in objList:
    data_x[feat] = le.fit_transform(data_x[feat].astype(str))
    
data_y = dataframe.loc[:][21]
data_y.unique()


 
train_x, test_x, train_y, test_y = train_test_split(data_x,data_y, train_size=0.7)
   
lr = linear_model.LogisticRegression()
lr.fit(train_x, train_y)
mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(train_x, train_y)

print ("Logistic regression Train Accuracy :: ", metrics.accuracy_score(train_y, lr.predict(train_x))) 
print ("Logistic regression Test Accuracy :: ", metrics.accuracy_score(test_y, lr.predict(test_x)))
y_pred = mul_lr.predict(test_x)
print ("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(train_y, mul_lr.predict(train_x)))
print ("Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(test_y, mul_lr.predict(test_x)))
confusion_matrix(test_y,y_pred )
accuracy_score(test_y,y_pred)


''' ANN METHOD'''


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from keras.utils import to_categorical


model = Sequential()
model.add(Dense(10, input_dim=21, activation='relu'))
model.add(Dense(23, activation='relu'))
model.add(Dense(23, activation='relu'))
model.add(Dense(17, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
model.fit(train_x,train_y, epochs= 500, verbose =0)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
y_pred = model.predict_classes(test_x)
confusion_matrix(test_y,y_pred )
accuracy_score(test_y,y_pred)
'''decision Tree'''


import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(train_x, train_y)
y_pred = classifier.predict(test_x)
classifier.score(test_x, test_y)

confusion_matrix(test_y,y_pred )
print(pd.crosstab(test_y, y_pred, rownames=['Actual Species'], colnames=['Predicted Species']))
print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(test_y, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, y_pred)))
accuracy_score(test_y,y_pred)