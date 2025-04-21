import pandas as pd

files = [
    'acc_1_5Q_1_C2L4',
    'acc_2_6Q_1_C2L4',
    'acc_2_6Q_2_C4L1',
]

train_cols = []  # 第3、4欄 (index 2, 3)
test_cols = []   # 第5、6欄 (index 4, 5)

# 讀取並擷取欄位
for f in files:
    df = pd.read_csv(f'acc/{f}', delim_whitespace=True, header=None)
    train_cols.append(df.iloc[:, [2, 3]])
    test_cols.append(df.iloc[:, [4, 5]])

# 合併所有 train / test 欄位
train_df = pd.concat(train_cols, axis=1)
test_df = pd.concat(test_cols, axis=1)

# 每份行數
chunk_size = 5

# 分割 train_df 成 5 份
for i in range(5):
    chunk = train_df.iloc[i * chunk_size:(i + 1) * chunk_size]
    chunk.to_csv(f"acc_comb/acc_train_{i+1}.csv", sep='\t', index=False, header=False)

# 分割 test_df 成 5 份
for i in range(5):
    chunk = test_df.iloc[i * chunk_size:(i + 1) * chunk_size]
    chunk.to_csv(f"acc_comb/acc_test_{i+1}.csv", sep='\t', index=False, header=False)

