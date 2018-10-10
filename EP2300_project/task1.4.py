import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

#read csv file into a pandas DataFrame named seperately
X = pd.read_csv(r"/Users/fanyuan/EP2300_jupyter notebook/data/X.csv")

fig = plt.figure('Task1(3)')
ax0 = fig.add_subplot(2,2,1)
ax1 = fig.add_subplot(2,2,3)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,4)
plt.style.use('ggplot')
data= pd.DataFrame(X)
mem_use = X["%%memused"]
cpu_uti = X["all_%%usr"]

#histogram plot generation with title and color parameters, etc.
#Memory Usage plot
num_bins = int(mem_use.quantile(0.01))
ax0.title.set_text('My %s-bin histogram plot for Memory usage'%num_bins)


ax0.hist(mem_use, bins = num_bins, edgecolor = 'k', color ='b', lw = 0.2, alpha = 0.5, label= 'frequency')


font_size = 15
ax0.set_xlabel('%%memused', fontsize=font_size)
ax0.set_ylabel('Frequency', fontsize=font_size)
ax0.legend()
ax0.tick_params(axis='both', which='major', labelsize=font_size-2)
ax0.tick_params(axis='both', which='minor', labelsize=font_size-4)


#density plot generation
#Memory usage plot
ax1.title.set_text('My density plot for Memory usage')


ax1 = sns.distplot(mem_use, color ='g', label= 'density', ax=ax1)



ax1.set_xlabel('%%memused', fontsize=font_size)
ax1.set_ylabel('Density', fontsize=font_size)
ax1.legend()
ax1.tick_params(axis='both', which='major', labelsize=font_size-2)
ax1.tick_params(axis='both', which='minor', labelsize=font_size-4)

#CPU utilization plot
#histogram plot generation with title and color parameters, etc.
num_bins_1 = int(cpu_uti.quantile(0.01))
ax2.title.set_text('My %s-bin histogram plot for CPU utilization'%num_bins_1)


ax2.hist(cpu_uti, bins = num_bins_1, edgecolor = 'b', color ='y', lw = 0.2, alpha = 0.5, label= 'frequency')


ax2.set_xlabel('all_%%usr', fontsize=font_size)
ax2.set_ylabel('Frequency', fontsize=font_size)
ax2.legend()
ax2.tick_params(axis='both', which='major', labelsize=font_size-2)
ax2.tick_params(axis='both', which='minor', labelsize=font_size-4)


#density plot generation
ax3.title.set_text('My density plot for CPU utilization')

ax3 = sns.kdeplot(cpu_uti, color ='r',shade = True, label= 'density',ax=ax3)



ax3.set_xlabel('all_%%usr', fontsize=font_size)
ax3.set_ylabel('Density', fontsize=font_size)
ax3.legend()
ax3.tick_params(axis='both', which='major', labelsize=font_size-2)
ax3.tick_params(axis='both', which='minor', labelsize=font_size-4)


plt.tight_layout()
plt.show()