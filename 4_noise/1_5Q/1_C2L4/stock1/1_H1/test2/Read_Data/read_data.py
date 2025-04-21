import os,sys
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from Global import *

def Read_data(data):
  fr=open("../../../../../../1_data/4_training_data/H%d/%dQ/%s_%d"%(Horizon,N_Qubit,data,Stock),"r")
  img=[]
  trg=[]
  for line in fr:
    lx=line.split()
    for i in range (len(lx)-1):
      img.append(float(lx[i]))
    trg.append(int(lx[len(lx)-1]))
  fr.close()
  img2=torch.tensor(img,dtype=Dtype).reshape(-1,2**N_Qubit)
  img3=img2/torch.sqrt(torch.sum(img2**2,dim=1,keepdim=True))
  del img,img2
  return trg,img3

def Read_Data():
  test_tar,test_img=Read_data("Test")
  test_tar_one=One_Hot(test_tar)
  train_tar,train_img=Read_data("Train")
  train_tar_one=One_Hot(train_tar)
  return test_tar,test_tar_one,test_img,train_tar,train_tar_one,train_img

def One_Hot(target):
  target_tensor = torch.tensor(target, dtype=torch.long) 
  one_hot_encoded = F.one_hot(target_tensor,3) 
  return one_hot_encoded.to(dtype=torch.float32) 

def Con_Unitary(par,s_ker):
  U,S,VT=torch.linalg.svd(par.reshape(s_ker,s_ker))
  Q=torch.matmul(U,VT)
  del U,S,VT
  return Q

def Correct(epoch,data,fw1,pred,target):
  count=0
  fw1.write("%-10s %6d\n"%(data,epoch))
  for i in range (len(target)):
    fw1.write("%3d %3d   "%(pred[i],target[i]))
    if i%10==9:
      fw1.write("\n")
    if pred[i]==target[i]:
      count=count+1
  if i%10!=9:
    fw1.write("\n")
  per=100*count/len(target)
  fw1.write("\n")
  fw1.flush()
  return per

def OutAcc(i,fw,acc_tran,acc_test):
  fw.write("%8d %7.2f %7.2f\n"%(i,acc_tran,acc_test))
  fw.flush()

