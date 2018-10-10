
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_validation import train_test_split



# Read X,Y traces into pandas DataFrames named X,Y

train_data= pd.read_csv(r'/Users/fanyuan/EP2300_jupyter notebook/data/X.csv',usecols =[0,1,2,3,4,5,6,7,8,9])
data_Y = pd.read_csv(r'/Users/fanyuan/EP2300_jupyter notebook/data/Y.csv')

#sample devide in Linear Regression
Time = pd.read_csv(r'/Users/fanyuan/EP2300_jupyter notebook/data/X.csv',usecols =['TimeStamp']).values
train_target = data_Y['DispFrames']


#test_size：样本占比，如果是整数的话就是样本的数量,random_state：是随机数的种子。
X_train,X_test, Y_train, Y_test = train_test_split(train_data,train_target,test_size=0.3,train_size = 0.7, random_state=1) 


#simulation
model = LinearRegression()
model.fit(X_train,Y_train)

np.set_printoptions(precision=2)
a = model.intercept_#截距
b = model.coef_#回归系数
print ("The model has coefficients of",b,"\n The offset is %.2f"%a)

#For test set data, predict with predictor function
Y_pred = model.predict(X_test)
Y_pred_all = model.predict(train_data)

# Compute the normalized mean absoluted error of the model over the test set
Y_train_AVG = np.mean(Y_train)
Y_naive = np.array(Y_train_AVG).repeat(1080)

Y_test_AVG = np.mean(Y_test.values)
print(Y_test_AVG)
mae_test=np.sum(np.absolute(Y_test-Y_pred))/len(Y_test)
mae_test_n=np.sum(np.absolute(Y_test-Y_naive))/len(Y_test)
NMAE=(mae_test)/Y_test_AVG
NMAE_naive=(mae_test_n)/Y_train_AVG
print("Average of training set in naive method is",Y_train_AVG)
print("Normalized Mean Absolute Error(NMAE): %.2f" %NMAE, "(",NMAE,")")
print("Normalized Mean Absolute Error(NMAE) in naive method: %.2f" %NMAE_naive, "(",NMAE_naive,")")


plt.figure()
plt.title("The Time series plot for 3 methods")
plt.plot(range(len(train_target)),Y_pred_all,'b',label="predict",alpha=0.7)
plt.plot(range(len(train_target)),train_target ,'r',label="test",alpha=0.7)
plt.plot(range(len(train_target)),[Y_train_AVG,]*3600 ,'k',label="naive ")
plt.legend(loc=0)
plt.xlabel("DataFrames")
plt.ylabel("DispFrames values")


plt.figure()
plt.plot(range(len(Y_pred)),Y_pred,'b',label="predict")
plt.plot(range(len(Y_pred)),Y_test,'r',label="test")
plt.legend(loc="upper right") #显示图中的标签
plt.xlabel("the number of sales")
plt.ylabel('value of sales')


#Time series polt shows measurements and estimations
plt.figure('task2_scatter')
plt.title("Scatter plot of estimations")
plt.scatter(range(len(Y_pred)),Y_pred,marker = '.',color = 'g',label="predict",alpha=0.5)
plt.scatter(range(len(Y_test)),Y_test ,marker = '+',color = 'r',label="test",alpha=0.7)
plt.legend(loc=0)
plt.xlabel("DataFrames")
plt.ylabel('value of video frame rates in estimations and measurements')




#density plot & histogram for DispFrames
fig=plt.figure('Task2.1(d)',figsize=(10,5))
ax0=fig.add_subplot(1,2,1)
ax1=fig.add_subplot(1,2,2)
plt.style.use('ggplot')
#density plot
ax0.title.set_text('My density plot for Video Frame Rate')


ax0 = sns.kdeplot(Y_test, color ='g', label= 'density', ax=ax0)

font_size = 15
ax0.set_xlabel('DispFrames', fontsize=font_size)
ax0.set_ylabel('density', fontsize=font_size)
ax0.legend(loc=0)
ax0.tick_params(axis='both', which='major', labelsize=font_size-2)
ax0.tick_params(axis='both', which='minor', labelsize=font_size-4)
#histogram plot
ax1.title.set_text('My 30-bin histogram plot for Video Frame Rate')
ax1.hist(Y_test, bins = 30, edgecolor = 'k', color ='b', lw = 0.2, alpha = 0.5, label= 'frequency')

ax1.set_xlabel('DispFrames', fontsize=font_size)
ax1.set_ylabel('frequency', fontsize=font_size)
ax1.legend(loc=0)
ax1.tick_params(axis='both', which='major', labelsize=font_size-2)
ax1.tick_params(axis='both', which='minor', labelsize=font_size-4)

#density plot of prediction errors
predict_err=Y_test-Y_pred
plt.figure('task2_1(e)')
plt.title("Density plot of prediction errors")
sns.distplot(predict_err, color ='orange', label= 'density')
font_size = 15
plt.xlabel('Error values', fontsize=font_size)
plt.ylabel('density', fontsize=font_size)
plt.legend(loc=0)
plt.tick_params(axis='both', which='major', labelsize=font_size-2)
plt.tick_params(axis='both', which='minor', labelsize=font_size-4)
plt.tight_layout()
plt.show()

