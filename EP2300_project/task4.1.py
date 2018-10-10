import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from scipy.special import comb
import time

start = time.time()
feature_cols=['runq-sz','%%memused','proc/s','cswch/s','all_%%usr','ldavg-1','totsck','pgfree/s','plist-sz','file-nr']
Subset_X_train=[]
NMAE_sets =[]


def findsubsets(S):
	results = [] 
	for subset_len in range(1,len(S)+1): 
		for combo in itertools.combinations(S, subset_len): 
			results.append(np.array(combo))
			
	return results

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


#Method 1 Optimal method (small number of features suits)

Subset_X_train =findsubsets(feature_cols)
for x in range(1,1024):
	NMAE_sets.append(NMAE_function(Subset_X_train[x-1]))


end = time.time()
#histogram plot

font_size = 15
plt.title('My 300-bin histogram plot for NMAE on each model')
plt.hist(NMAE_sets, bins = 300, edgecolor = 'k', color ='b', lw = 0.5, alpha = 0.5, label= 'frequency')
plt.xlabel('NMAE values', fontsize=font_size)
plt.ylabel('frequency', fontsize=font_size)
plt.legend(loc=0)
plt.tick_params(axis='both', which='major', labelsize=font_size-2)
plt.tick_params(axis='both', which='minor', labelsize=font_size-4)

#finding the smallest NMAE model subset
feature_min= NMAE_sets.index(min(NMAE_sets))
print("The features subset model with smallest NMAE is : \n", Subset_X_train[feature_min])



#box plot for NMAE of models contain 1-10 number of features, in silly way TUT
	
A=NMAE_sets[:9]
B=NMAE_sets[10:54]
C=NMAE_sets[55:174]
D=NMAE_sets[175:384]
E=NMAE_sets[385:636]
F=NMAE_sets[637:846]
G=NMAE_sets[847:966]
H=NMAE_sets[967:1011]
I=NMAE_sets[1012:1021]
J=NMAE_sets[1022]


data_to_plot = [A, B, C, D, E, F, G, H, I, J]

plt.figure("Task4_box")

plt.ylabel('NMAE values', fontsize = font_size)
plt.xlabel('Number of Features', fontsize = font_size)
plt.boxplot(data_to_plot,sym = "o")
plt.tight_layout()
plt.show()


print("execute time of Method 1 is :", (end-start)," seconds")
