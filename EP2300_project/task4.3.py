
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest,SelectPercentile
import time



	


def NMAE_function(i):
	linear_model= Lasso(alpha = i)
	model=linear_model.fit(X_train,Y_train)
	Y_pred=linear_model.predict(X_test)
	# the y_average should be compute based on the y(i) of test set
	Y_avg=np.mean(Y_test.values)
	mae_test=np.sum(np.absolute(Y_pred-Y_test))/len(Y_test)
	NMAE=(mae_test)/Y_avg
	x= np.sum(model.coef_ != 0)

	return NMAE,x


start = time.time()
feature_cols=['runq-sz','%%memused','proc/s','cswch/s','all_%%usr','ldavg-1','totsck','pgfree/s','plist-sz','file-nr']
# Read X,Y traces into pandas DataFrames named X,Y

train_data= pd.read_csv(r'/Users/fanyuan/EP2300_jupyter notebook/data/X.csv',usecols=feature_cols)
data_Y = pd.read_csv(r'/Users/fanyuan/EP2300_jupyter notebook/data/Y.csv')
train_target = data_Y['DispFrames']
#sample devide in Linear Regression
X_train,X_test,Y_train,Y_test= train_test_split(train_data,train_target,test_size=0.3,random_state=1)


alphas=10**np.linspace(-1,3)
Lasso_set=pd.DataFrame(alphas,columns=['alpha'])
Lasso_set['NMAE']=Lasso_set.alpha.apply(lambda x : NMAE_function(x)[0])
Lasso_set['None_0_features']=Lasso_set.alpha.apply(lambda x : NMAE_function(x)[1])
print(Lasso_set)
end = time.time()


plt.figure("Task4_3")
plt.plot(alphas,Lasso_set['None_0_features'],'r',label="feature != 0",alpha=0.7)
plt.xlabel("number of alpha")
plt.xscale('log')
plt.ylabel("feature values")
plt.legend(loc=0)


print("execute time of Method 3 is :", (end-start)," seconds")
plt.tight_layout()
plt.show()