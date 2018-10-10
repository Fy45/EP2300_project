
cpu_mem_smaller_num = X.loc[(X['%%memused'] < 50) & (X['all_%%usr'] < 90)].TimeStamp.count()
print("The number of observations with CPU utilization smaller than 90% and memory utilization smaller than 50% is:",cpu_mem_smaller_num)

cswch_ls_60000 = X[X['cswch/s'] < 60000]
print("The average number of used sockets for observations with less than 60000 context switches per seconds is",np.round(cswch_ls_60000["totsck"].mean(),decimals=2))
