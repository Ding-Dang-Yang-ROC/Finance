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
    return data.to_numpy()

def OutPut(data,t):
  d1=np.mean(data[:,:-5],axis=1)
  m=0
  fw=open('data/%s_%d.txt'%(t,m),'w')
  for i in range (d1.shape[0]-1):
    for j in range (data.shape[1]-5):
      fw.write('%10.7e '%(data[i][j]))
    for j in range (data.shape[1]-5,data.shape[1]):
      fw.write('%1d '%(data[i][j]-1))
    fw.write('\n')
    if np.abs(d1[i+1]-d1[i]) > 0.06:
      fw.close()
      m=m+1
      fw=open('data/%s_%d.txt'%(t,m),'w')
  for j in range (data.shape[1]-5):
    fw.write('%10.7e '%(data[d1.shape[0]-1][j]))
  for j in range (data.shape[1]-5,data.shape[1]):
    fw.write('%1d '%(data[d1.shape[0]-1][j]-1))
  fw.close()

def Main():
  for i in range (2):
    data=read_data("../2_top_40/data/%s.txt"%(t[i]))
    OutPut(data,t[i])

Main()

