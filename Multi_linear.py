# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 02:20:33 2020

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
 
print ("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(train_y, mul_lr.predict(train_x)))
print ("Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(test_y, mul_lr.predict(test_x)))

coeff = mul_lr.coef_