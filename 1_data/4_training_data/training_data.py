import numpy as np
import pandas as pd

Num_Qbit=[5,6,8]
Hor=[1,2,3,5,10]
t=['Test','Train']

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
    data = pd.DataFrame(data_list).to_numpy()
    data2=data.reshape(-1,45)
    data3=data2[:,:40].reshape(-1,10,2,2)
    data4=data3[:,:8,:,0].reshape(-1,16)
    data5=data3[:,:8,:,1].reshape(-1,16)
    data6=data2[:,40:]
    return data4,data5,data6

def Comb(data):
  n1=data.shape[0]-window-Horinzon+1
  data2=np.zeros((n1,2**(Num_qbit-1)))
  for i in range (n1): # n1
    d2=data[i:i+window,:].reshape(-1)
    d3=np.min(d2)
    d4=d2-d3
    row_norm=np.linalg.norm(d4, keepdims=True)
    if row_norm==0:
       N = d4.size
       data2[i]=np.ones_like(d4)/np.sqrt(N)
    else:
       data2[i]=d4/row_norm
  return data2

def OutPut(t,p,h,data1,data2,tar):
  fw=open('H%d/%dQ/%s_%d'%(Hor[h],Num_qbit,t,p),'w')
  for i in range (data1.shape[0]): # data1.shape[0]
    for j in range (2**(Num_qbit-1)):
      fw.write('%10.7f '%(data1[i,j]))
    for j in range (2**(Num_qbit-1)):
      fw.write('%10.7f '%(data2[i,j]))
    for j in range (h,h+1):
      fw.write('%1d '%(tar[i,j]))
    fw.write('\n')
  fw.close()

def Main():
  global Num_qbit,window,Horinzon
  for h in range (5): # 5
    Horinzon=Hor[h]
    for q in range (3): # 3
      Num_qbit=Num_Qbit[q]
      window=2**(Num_qbit-5)
      for i in range (5): # 5
        for j in range (2): # 2
          print(h,q,i,j)
          data1,data2,tar=read_data("../3_5_stock/data/%s_%d.txt"%(t[j],i))
          data3=Comb(data1)
          data4=Comb(data2)
          OutPut(t[j],i,h,data3,data4,tar)

Main()


