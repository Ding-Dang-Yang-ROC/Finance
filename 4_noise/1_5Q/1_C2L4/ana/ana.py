import numpy as np
import re

def Read_Acc(filename):
  with open(filename,'r') as f:
    lines=f.readlines()
  data=lines[1].split()
  acc_train=float(data[1])
  acc_test=float(data[2])
  return acc_train,acc_test

Acc_train=[]
Acc_test=[]
data1s=['stock1','stock2','stock3','stock4','stock5']
data2s=['1_H1','2_H2','3_H3','4_H5','5_H10']
data3s=['test1','test2','test3','test4','test5']

for data1 in data1s:
  for data2 in data2s:
    for data3 in data3s:
      filename = f"../{data1}/{data2}/{data3}/Result/accu_sum"
      acc_train,acc_test=Read_Acc(filename)
      Acc_train.append(acc_train)
      Acc_test.append(acc_test)

Acc_trn=np.array(Acc_train).reshape(5,5,5) #.min(axis=3)
Acc_ten=np.array(Acc_test).reshape(5,5,5)

Acc_trnm=np.mean(Acc_trn,axis=2)
Acc_trnd=np.std(Acc_trn,axis=2,ddof=1)

Acc_tenm=np.mean(Acc_ten,axis=2)
Acc_tend=np.std(Acc_ten,axis=2,ddof=1)

with open("acc_1_5Q_1_C2L4_noise", "w") as file:
  for i in range(5):
    for j in range(5):
      file.write('%6s %-5s %6.2f %6.2f   %6.2f %6.2f\n'%(data1s[i],data2s[j],Acc_trnm[i][j],Acc_trnd[i][j],Acc_tenm[i][j],Acc_tend[i][j]))

