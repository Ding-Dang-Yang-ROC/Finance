import pandas as pd
import numpy as np

filenames = [f'../../1_merge/acc_noise_comb/acc_noise_train_{i}.csv' for i in range(1, 6)]

data_list = [pd.read_csv(fname, header=None, delim_whitespace=True) for fname in filenames]

stacked = np.stack([df.values for df in data_list])
average = np.mean(stacked, axis=0)

np.savetxt('acc_noise_train_6.csv', average, fmt='%.2f', delimiter='\t')

