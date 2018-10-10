import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from pandas import Series,DataFrame
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


#import the data and use pandas to read into a dataframe
raw = pd.read_csv(r'/Users/fanyuan/EP2300_jupyter notebook/data/Y.csv')
train_data= pd.read_csv(r'/Users/fanyuan/EP2300_jupyter notebook/data/X.csv',usecols =[0,1,2,3,4,5,6,7,8,9])
Time = pd.read_csv(r'/Users/fanyuan/EP2300_jupyter notebook/data/X.csv',usecols =['TimeStamp']).values
train_target = raw['DispFrames']

#data split with train70% test30%
X_train,X_test, Y_train, Y_test = train_test_split(train_data,train_target,test_size=0.3,train_size = 0.7, random_state=1) 

#using Linear regression for evaluation
model= LinearRegression()
model.fit(X_train,Y_train)
np.set_printoptions(precision=2)
Y_pred = model.predict(X_test)


def findSLA(x,y):
	if x > y:
		return 1
	else:
		return 0
	pass

Y_pred = pd.DataFrame(Y_pred,columns=['SLA']).SLA.apply(lambda x: findSLA(x,18))
Y_test = pd.DataFrame(Y_test).DispFrames.apply(lambda x:findSLA(x,18))
extend_cm = confusion_matrix(Y_test,Y_pred)
TN, FP, FN, TP = extend_cm.ravel()
extend_ERR = 1 - (TP+TN)/len(Y_test)

print("The extended new classifier has classification error of : %.2f"%extend_ERR,"(",extend_ERR,")")