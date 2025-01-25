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
    return data.to_numpy()[:,-5:]

def Count(data):
  t=[]
  for i in range (3):
    t.append(np.sum(data[:,0]==i)/data.shape[0])
  print("%7d %6.3f %6.3f %6.3f %6.3f"%(data.shape[0],t[0],t[1],t[2],t[0]+t[1]+t[2]))

def Main():
  for i in range (5): # 5
    print(i)
    for j in range (2): # 2
      data=read_data("../data/%s_%d.txt"%(t[j],i))
      Count(data)

Main()

