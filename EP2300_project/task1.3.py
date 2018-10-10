
import pandas as pd
import numpy as np
from pandas import DataFrame


#read csv file into a pandas DataFrame named seperately
X = pd.read_csv(r'/Users/fanyuan/Downloads/X.csv')

# Convert TimeStamp into date-time format
timeIndex=pd.to_datetime(X['TimeStamp'], unit='s')
X.index=timeIndex

import matplotlib.pyplot as plt

# Generate a figure with one subplot 
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

# Produce the plots for specified features sets
mem = X['%%memused'].plot(color='r')
cpu = X['all_%%usr'].plot(color='b')
axes.grid(True)

# Customizing plot 
font_size = 15
plt.title('Time series of memory usage and CPU utilization')
plt.xlabel('Timestamp', fontsize=font_size)
plt.ylabel('Values', fontsize=font_size)
plt.legend(('%%memused','all_%%usr'), loc = 0, shadow=False, fancybox=False, fontsize=font_size-4)
plt.tick_params(axis='both', which='major', labelsize=font_size-2)
plt.tick_params(axis='both', which='minor', labelsize=font_size-4)





#Producing the box plot for both features
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
mem_use = X["%%memused"]
cpu_uti = X["all_%%usr"]
data = pd.DataFrame({"Memory_usage":mem_use,
                     "CPU_utilization":cpu_uti})

plt.title('Box plot of memory usage and CPU utilization')

data.boxplot()
plt.ylabel('Values', fontsize = font_size)
plt.xlabel('Feature', fontsize = font_size)
plt.tight_layout()
plt.show()