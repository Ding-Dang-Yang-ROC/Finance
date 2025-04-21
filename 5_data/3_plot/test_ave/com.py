import numpy as np

# 讀取資料
file1 = "data/acc_test_6.csv"
file2 = "data/acc_noise_test_6.csv"

data1 = np.loadtxt(file1)
data2 = np.loadtxt(file2)

# 擷取所需欄位並合併
qcnn_1 = np.hstack([data1[:, 0:2], data2[:, 0:2]])
qcnn_2 = np.hstack([data1[:, 2:4], data2[:, 2:4]])
qcnn_3 = np.hstack([data1[:, 4:6], data2[:, 4:6]])

# 儲存成 CSV 檔
np.savetxt("data_combine/QCNN_1.csv", qcnn_1, fmt="%.2f", delimiter="\t")
np.savetxt("data_combine/QCNN_2.csv", qcnn_2, fmt="%.2f", delimiter="\t")
np.savetxt("data_combine/QCNN_3.csv", qcnn_3, fmt="%.2f", delimiter="\t")

