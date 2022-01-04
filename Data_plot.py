# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 00:30:02 2020

@author: Mypc
"""

from scipy.stats import chi2_contingency 

import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from scipy.stats import chi2_contingency 
from sklearn.metrics import roc_curve, auc
dataframe = pd.read_excel('Arko shankar.xlsx')
from sklearn.preprocessing import label_binarize
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
unique_list =[]
for i in range (dataframe.shape[1]):
    unique_list.append(dataframe.iloc[:,i].unique())
    
df = pd.DataFrame(unique_list)

dataframe.iloc[:,21].value_counts()
last_col = []
for i in range(len(df.iloc[21,:])):
    last_col.append(df.iloc[21,i])
dataframe[dataframe.iloc[:,21]==14].iloc[:,0].value_counts()

data = [[143,37,11,66,0,0],[26,142,43,0,11,2],[0,3,35,0,10,4]]
stat,p,dof,expected = chi2_contingency(data)

data = [[207, 282, 241], [234, 242, 232]] 
stat, p, dof, expected = chi2_contingency(data)


'''LIN'''
import pandas as pd
from pandas import DataFrame

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

y = label_binarize(data_y, classes=[6,8,10,12,14,16])
n_classes = y.shape[1]
 
train_x, test_x, train_y, test_y = train_test_split(data_x,data_y, train_size=0.6)
t_y = label_binarize(test_y, classes=[6,8,10,12,14,16])
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
y_pred_df = pd.DataFrame(y_pred)

y_p = label_binarize(y_pred, classes=[6,8,10,12,14,16])
test_x_df = pd.DataFrame(test_x)
test_x_df.reset_index(inplace = True)

overall = []
d = [14,12,10,16,8,6]
rel = []
for j in range(1,21):
    rel=[]
    for i in d:
        rel.append(test_x_df[y_pred_df.iloc[:,0]==i].iloc[:,j].value_counts())
    overall.append(rel)
size = 5
final =[]
for i in overall[17]:
    pred=[0 for i in range(size)]
    data_top = i.head()
    ls = list(data_top.index)
    ls.sort()
    for j in ls:
        pred[j]=i[j];
    final.append(pred)  

#data =final
#stat,p,dof,expected = chi2_contingency(data)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(t_y[:, i], y_p[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(t_y.ravel(), y_p.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','green','red','yellow'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()



# ANN 
