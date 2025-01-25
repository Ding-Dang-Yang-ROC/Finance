import numpy as np
import pandas as pd

t=['Train','Test']

def read_data(path):
    data_list = []
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            d_str = line.split()
            d_tem = [float(d) for d in d_str]
            data_list.append(d_tem)
    data = pd.DataFrame(data_list)
    return data.T.to_numpy()[:,:40],data.T.to_numpy()[:,-5:]

def OutPut(data1,data2,t):
  fw=open("data/%s.txt"%(t),'w')
  for i in range (data1.shape[0]):
    for j in range (data1.shape[1]):
      fw.write('%10.7e '%(data1[i][j]))
    for j in range (data2.shape[1]):
      fw.write('%1d '%(data2[i][j]))
    fw.write('\n')

def Main():
  for i in range (2): # 2
    data1,data2=read_data("../1_ori_data/%s_Dst_NoAuction_MinMax_CF_9.txt"%(t[i]))
    OutPut(data1,data2,t[i])

Main()

