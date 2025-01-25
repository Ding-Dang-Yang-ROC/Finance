# Quantum Convolutional Neural Networks for Stock Price Prediction

This repository contains the implementation of Quantum Convolutional Neural Networks (QCNNs) for stock price prediction, leveraging the FI-2010 dataset. The project is organized into multiple directories to handle data preprocessing, model training, and testing. Follow the steps below to set up the environment and explore the code.

## 1. Software Installation

Before running the code, install the required Python libraries:

pip install numpy pandas torch torchvision pennylane
Ensure that your Python version is compatible with these libraries.

## 2. Directory Structure

### `1_data`: Download and preprocess the FI-2010 dataset.
- **`1_data/1_ori_data`**:  
  - Download the **FI-2010 dataset** files `Test_Dst_NoAuction_MinMax_CF_9.txt` and `Train_Dst_NoAuction_MinMax_CF_9.txt` into this directory.  
  - Dataset download URL: [FI-2010 Dataset on Etsin](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649).

- **`1_data/2_top_40/data`**:  
  - Extract the **first 40 rows** and the **last 5 rows** from the original dataset.

- **`1_data/3_5_stock`**:  
  - Split the data into **5 individual files**, each corresponding to one stock.

- **`1_data/4_training_data`**:  
  - Convert the processed data into **training and testing datasets**.

### `2_train`: Training and testing programs.
- **`2_train/1_5Q`**:  
  - Code for training and testing using **5 qubits** (refer to Fig. 1(a) in the paper).

- **`2_train/2_6Q`**:  
  - Code for training and testing using **6 qubits**.  
  - Subdirectories:
    - **`2_train/2_6Q/1_C2L4`**: Implements the **Fig. 1(b)** configuration with **2 qubits and 4 layers**.  
    - **`2_train/2_6Q/3_C4L1`**: Implements the **Fig. 1(c)** configuration with **4 qubits and 1 layer**.
