import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

for idx in range(1, 4):
    filename = f'data_combine/QCNN_{idx}.csv'
    if not os.path.exists(filename):
        print(f'{filename} 不存在，略過')
        continue

    # 讀取資料
    df = pd.read_csv(filename, header=None, sep='\t')
    means = df.iloc[:, [0, 2]].values
    stds = df.iloc[:, [1, 3]].values

    group_count, bar_count = means.shape
    x = np.arange(group_count)
    width = 0.25
    offsets = [-width, 0, width]
    group_labels = ['1', '2', '3', '5', '10']

    # 繪圖
    plt.figure(figsize=(10, 6))
    for i in range(bar_count):
        plt.bar(x + offsets[i], means[:, i], yerr=stds[:, i],
                width=width, capsize=5)

    plt.xticks(x, group_labels, fontsize=20)
    plt.yticks(fontsize=20)

    plt.tight_layout()
    outname = f'png/QCNN_{idx}.png'
    plt.savefig(outname, dpi=300)
    plt.close()
    print(f'已儲存：{outname}')

