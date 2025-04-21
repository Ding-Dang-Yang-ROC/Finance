import os,sys
import torch
import numpy as np
import pennylane as qml
from Global import *
from Read_Data.read_data import *

dev = qml.device("default.mixed", wires=N_Qubit)

def apply_noise_all_qubits(dep=0.05, phase=0.03):
    for i in range(N_Qubit):
        qml.DepolarizingChannel(dep, wires=i)
        qml.PhaseDamping(phase, wires=i)

@qml.qnode(dev, interface="torch")
def quantum_circuit(state, cm0, cm1, cm2, cm3):
    qml.QubitStateVector(state, wires=range(N_Qubit))
    qml.QubitUnitary(cm0, wires=[0, 5])
    apply_noise_all_qubits()
    qml.QubitUnitary(cm1, wires=[0, 4])
    apply_noise_all_qubits()
    qml.QubitUnitary(cm2, wires=[0, 3])
    apply_noise_all_qubits()
    qml.QubitUnitary(cm3, wires=[0, 2])
    apply_noise_all_qubits()
    return qml.probs(wires=[0, 1])

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

def Read_Par():
  with open(f'../../../../../../2_train/2_6Q/1_C2L4/{stock}/{Dir}_H{Horizon}/{Test}/Result/par',"r") as f:
    data=[float(num) for line in f for num in line.strip().split()]
  par1=torch.tensor(data[:N_Lay1*N_par1],dtype=torch.float32).reshape(N_Lay1,N_par1)
  par2=torch.tensor(data[N_Lay1*N_par1:],dtype=torch.float32).reshape(N_par2,3)
  return par1,par2

def Main():
  Par1,Par2=Read_Par()
  test_tar,test_one,test_img,train_tar,train_one,train_img=Read_Data()
  fw1=open("Result/accu_case_ori","w")
  fw2=open("Result/accu_sum","w")
  fw2.write("   epoch   train    test\n")
  pdig_train_pen=Pred_Pen(train_img,Par1.detach(),Par2.detach())
  pdig_test_pen=Pred_Pen(test_img,Par1.detach(),Par2.detach())
  acc_tran_pen=Correct(1000,"train",fw1,pdig_train_pen,train_tar)
  acc_test_pen=Correct(1000,"test",fw1,pdig_test_pen,test_tar)
  OutAcc(1000,fw2,acc_tran_pen,acc_test_pen)
  fw1.close()
  fw2.close()

Main()

