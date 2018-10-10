
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_validation import train_test_split


train_size =[50,100,200,500,1000,2520]
# Read X,Y traces into pandas DataFrames named X,Y

train_data= pd.read_csv(r'/Users/fanyuan/EP2300_jupyter notebook/data/X.csv',usecols =[0,1,2,3,4,5,6,7,8,9])
data_Y = pd.read_csv(r'/Users/fanyuan/EP2300_jupyter notebook/data/Y.csv')

#sample devide in Linear Regression
Time = pd.read_csv(r'/Users/fanyuan/EP2300_jupyter notebook/data/X.csv',usecols =['TimeStamp']).values
train_target = data_Y['DispFrames']

X_train,X_test, Y_train, Y_test = train_test_split(train_data,train_target,test_size=0.3,train_size = 0.7, random_state=1) 

#convert NMAE calculation into function
def NMAE_function(size,x,y):
	x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=(2520-size),random_state=1)
	linear_model= LinearRegression()
	model=linear_model.fit(x_train,y_train)
	y_pred=linear_model.predict(X_test)
	# the y_average should be compute based on the y(i) of test set
	y_avg=np.mean(Y_test.values)
	mae_test=np.sum(np.absolute(y_pred-Y_test))/len(Y_test)
	NMAE=(mae_test)/y_avg
	print ("Normalized Mean Absolute Error(NMAE) for train size: ",size, "is : %.2f" %NMAE, "(",NMAE,")")
	return NMAE


#six training sets

NList = np.random.randn(6)
print("The six training sets are: ",NList)
for i in range(len(train_size)):
    NList[i]=NMAE_function(train_size[i],X_train,Y_train)



#train the models for 50 different subsets

def NMAE_50(size,x,y,times):
	x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=(2520-size),random_state=times)
	linear_model= LinearRegression()
	model=linear_model.fit(x_train,y_train)
	y_pred=linear_model.predict(X_test)
	y_avg=np.mean(Y_test.values)
	mae_test=np.sum(np.absolute(y_pred-Y_test))/len(Y_test)
	NMAE=(mae_test)/y_avg
	return NMAE

#perform 50 times	
A=np.random.randn(50)
B=np.random.randn(50)
C=np.random.randn(50)
D=np.random.randn(50)
E=np.random.randn(50)
F=np.random.randn(50)
for x in range(50):
	for i in range(len(train_size)):
		NList[i]=NMAE_50(train_size[i],X_train,Y_train,x)
		A[x]=NList[0]
		B[x]=NList[1]
		C[x]=NList[2]
		D[x]=NList[3]
		E[x]=NList[4]
		F[x]=NList[5]



NList={"50 " : A, "100 " : B,
       "200 " : C, "500 " : D,
       "1000 " : E, "2520 " : F }



data=pd.DataFrame(NList)
plt.figure('task2_box')

data.boxplot()
 
plt.title('Boxplot of different training set observations')


plt.tight_layout()
plt.show()
