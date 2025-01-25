import os,sys
import torch
import numpy as np
import random
import datetime
import pennylane as qml
from torch import optim
from Global import *
from Read_Data.read_data import *

dev=qml.device("default.qubit",wires=N_Qubit)

def Para():
  par1=torch.rand((N_Lay1,N_par1),dtype=Dtype,requires_grad=True)
  par2=torch.rand((N_par2,3),dtype=Dtype,requires_grad=True)
  return par1,par2

def Train_Pyt(one,img):
  sam=one.shape[0]
  n_batch=int(np.ceil(sam/Batch))
  num=np.arange(sam)
  random.shuffle(num)
  for i in range(n_batch):
    beg_frm=i*Batch
    end_frm=min((i+1)*Batch,sam)
    batch=end_frm-beg_frm
    one_b=one[num[beg_frm:end_frm]]

    #  0 1234 5
    # 0 5 123 4
    # 0 4 12 3 5
    # 0 3 1 2 45
    # 0 2 1 345
    # 01 2345

    img1=img[num[beg_frm:end_frm]].reshape(batch,2,16,2).permute(1,3,2,0).reshape(4,16*batch) # 05 1234
    pred=torch.matmul(Con_Unitary(Par1[0],S_Ker1),img1).reshape(2,2,8,2,batch).permute(0,3,2,1,4).reshape(4,16*batch) # 04 1235
    pred=torch.matmul(Con_Unitary(Par1[1],S_Ker1),pred).reshape(2,2,4,2,2,batch).permute(0,3,2,1,4,5).reshape(4,16*batch) # 03 1245
    pred=torch.matmul(Con_Unitary(Par1[2],S_Ker1),pred).reshape(2,2,2,2,4,batch).permute(0,3,2,1,4,5).reshape(4,16*batch) # 02 1345
    pred=torch.matmul(Con_Unitary(Par1[3],S_Ker1),pred).reshape(2,2,2,8,batch).permute(4,0,2,1,3).reshape(batch,4,16) # 01 2345
    pred=pred.pow(2).sum(dim=2)
    err=torch.matmul(pred,Par2)-one_b
    loss=torch.square(err).sum()
    Opt.zero_grad()
    loss.backward()
    Opt.step()

def Pred_Pyt(img,par1,par2):
  pdig=[]
  sam=img.shape[0]
  n_batch=int(np.ceil(sam/Batch))
  num=np.arange(sam)
  cm0=Con_Unitary(par1[0],S_Ker1)
  cm1=Con_Unitary(par1[1],S_Ker1)
  cm2=Con_Unitary(par1[2],S_Ker1)
  cm3=Con_Unitary(par1[3],S_Ker1)
  for i in range(n_batch):
    beg_frm=i*Batch
    end_frm=min((i+1)*Batch,sam)
    batch=end_frm-beg_frm
    img1=img[num[beg_frm:end_frm]].reshape(batch,2,16,2).permute(1,3,2,0).reshape(4,16*batch) # 05 1234
    pred=torch.matmul(cm0,img1).reshape(2,2,8,2,batch).permute(0,3,2,1,4).reshape(4,16*batch) # 04 1235
    pred=torch.matmul(cm1,pred).reshape(2,2,4,2,2,batch).permute(0,3,2,1,4,5).reshape(4,16*batch) # 03 1245
    pred=torch.matmul(cm2,pred).reshape(2,2,2,2,4,batch).permute(0,3,2,1,4,5).reshape(4,16*batch) # 02 1345
    pred=torch.matmul(cm3,pred).reshape(2,2,2,8,batch).permute(4,0,2,1,3).reshape(batch,4,16) # 01 2345
    pred=pred.pow(2).sum(dim=2)
    pred=torch.matmul(pred,par2)
    dig=torch.argmax(pred,dim=1).to("cpu").tolist()
    pdig=pdig+dig
  del cm0,cm1,cm2,cm3
  return pdig

@qml.qnode(dev,interface="torch")
def quantum_circuit(state,cm0,cm1,cm2,cm3):
  qml.QubitStateVector(state,wires=range(N_Qubit))
  qml.QubitUnitary(cm0,wires=[0,5])
  qml.QubitUnitary(cm1,wires=[0,4])
  qml.QubitUnitary(cm2,wires=[0,3])
  qml.QubitUnitary(cm3,wires=[0,2])
  return qml.probs(wires=[0,1])

def Pred_Pen(img,par1,par2):
  pdig=[]
  cm0=Con_Unitary(par1[0],S_Ker1)
  cm1=Con_Unitary(par1[1],S_Ker1)
  cm2=Con_Unitary(par1[2],S_Ker1)
  cm3=Con_Unitary(par1[3],S_Ker1)
  for i in range(img.shape[0]):
    pred=quantum_circuit(img[i].cpu().numpy(),cm0,cm1,cm2,cm3)
    pred=pred.clone().to(dtype=Dtype)
    pred=torch.matmul(pred,par2)
    dig=torch.argmax(pred)
    pdig.append(dig)
  del cm0,cm1,cm2,cm3
  return pdig

def Main():
  global Par1,Par2,Opt
  Par1,Par2=Para()
  test_tar,test_one,test_img,train_tar,train_one,train_img=Read_Data()
  Opt=optim.SGD([Par1,Par2],lr=learning_rate,momentum=0.9)
  fw1=open("Result/accu_case_ori","w")
  fw2=open("Result/accu_sum","w")
  fw2.write("   epoch   train    test\n")
  for i in range (N_Ite):
    Train_Pyt(train_one,train_img)
    if i%10==9:
      pdig_train_pyt=Pred_Pyt(train_img,Par1.detach(),Par2.detach())
      pdig_test_pyt=Pred_Pyt(test_img,Par1.detach(),Par2.detach())
      acc_tran=Correct(i,"train",fw1,pdig_train_pyt,train_tar)
      acc_test=Correct(i,"test",fw1,pdig_test_pyt,test_tar)
      OutAcc(i,fw2,acc_tran,acc_test)
  pdig_train_pen=Pred_Pen(train_img,Par1.detach(),Par2.detach())
  pdig_test_pen=Pred_Pen(test_img,Par1.detach(),Par2.detach())
  acc_tran_pen=Correct(1000,"train",fw1,pdig_train_pen,train_tar)
  acc_test_pen=Correct(1000,"test",fw1,pdig_test_pen,test_tar)
  OutAcc(1000,fw2,acc_tran_pen,acc_test_pen)
  OutPara(Par1.detach(),Par2.detach())
  fw1.close()
  fw2.close()

Main()

