import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest,SelectPercentile
import time

start = time.time()
feature_cols=['runq-sz','%%memused','proc/s','cswch/s','all_%%usr','ldavg-1','totsck','pgfree/s','plist-sz','file-nr']
square_corr=pd.DataFrame(feature_cols,columns=['features'])
NMAE_set =[]

def findcorrcoefsqr(i):
	sample_corr = np.corrcoef(X_train[i],Y_train)
	result= sample_corr[0][1]**2
	return result



def NMAE_function(train_data):
	linear_model= LinearRegression()
	model=linear_model.fit(X_train[train_data],Y_train)
	Y_pred=linear_model.predict(X_test[train_data])
	# the y_average should be compute based on the y(i) of test set
	Y_avg=np.mean(Y_test.values)
	mae_test=np.sum(np.absolute(Y_pred-Y_test))/len(Y_test)
	NMAE=(mae_test)/Y_avg
	#print ("Normalized Mean Absolute Error(NMAE) for train size: ",size, "is : %.2f" %NMAE, "(",NMAE,")")
	return NMAE


# Read X,Y traces into pandas DataFrames named X,Y

train_data= pd.read_csv(r'/Users/fanyuan/EP2300_jupyter notebook/data/X.csv',usecols=feature_cols)
data_Y = pd.read_csv(r'/Users/fanyuan/EP2300_jupyter notebook/data/Y.csv')
train_target = data_Y['DispFrames']
#sample devide in Linear Regression
X_train,X_test,Y_train,Y_test= train_test_split(train_data,train_target,test_size=0.3,random_state=1)



#Method 2 (Heuristic method)
square_corr['square_corr']=square_corr.features.apply(lambda x : findcorrcoefsqr(x))
rank = square_corr.sort_values(by='square_corr',ascending = False)
print(rank)
for i in range(1,len(feature_cols)+1):
	feature_set = rank.head(i)['features']
	NMAE_set.append(NMAE_function(feature_set))
end = time.time()



#plot the error values with k features
plt.figure("Task4_2_1")
plt.plot(range(1,11),NMAE_set,'g',label="NMAE values",alpha=0.7)
plt.xlabel("Feature set with k features")
plt.xlim(1,11,1)
plt.ylabel("NMAE values")
plt.legend(loc=0)


#heat map
df= pd.concat( [train_data,train_target], axis=1 ).corr()
data_to_plot = pd.DataFrame(df, index=feature_cols, columns=feature_cols)

plt.figure("Task4_2_2")
sns.heatmap(data_to_plot,annot=True, vmax=1, square=True, cmap="plasma")




plt.tight_layout()
plt.show()