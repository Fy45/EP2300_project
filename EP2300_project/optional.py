import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor


# Read X,Y traces into pandas DataFrames named X,Y
features_cols=['runq-sz','%%memused','proc/s','cswch/s','all_%%usr','ldavg-1','totsck','pgfree/s','plist-sz','file-nr']
train_data= pd.read_csv(r'/Users/fanyuan/EP2300_jupyter notebook/data/X.csv',usecols=features_cols)
data_Y = pd.read_csv(r'/Users/fanyuan/EP2300_jupyter notebook/data/Y.csv')
Time = pd.read_csv(r'/Users/fanyuan/EP2300_jupyter notebook/data/X.csv',usecols =['TimeStamp']).values
train_target = data_Y['DispFrames']
#sample devide in Linear Regression
X_train,X_test,Y_train,Y_test= train_test_split(train_data,train_target,test_size=0.3,random_state=1)


regressor = RandomForestRegressor(oob_score=True, random_state=1)
regressor.fit(X_train,Y_train)

Y_pred = regressor.predict(X_test)
Y_train_average=np.mean(Y_train)
Y_test_average=np.mean(Y_test)
mae_test=np.sum(np.absolute(Y_test-Y_pred))/len(Y_test)
NMAE=(mae_test)/Y_test_average
print("Normalized Mean Absolute Error(NMAE) for Random Forest Regression: %.2f" %NMAE, "(",NMAE,")")

#Time series plot
plt.figure('opt_scatter')
plt.title("Scatter plot of estimations")
plt.scatter(range(len(Time)),np.array(Y_pred),marker = '.',color = 'red',label="predict",alpha=0.7)
plt.scatter(Time,np.array(Y_test) ,marker = '+',color = 'green',label="test",alpha=0.7)
plt.scatter(Time,np.array([Y_train_average,]*1080) ,marker = '-',color = 'blue',label="naive average")
plt.tight_layout()
plt.show()