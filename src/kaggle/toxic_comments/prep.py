import pandas as pd
import numpy as np

TEST_SIZE = 0.2

PWD = 'kaggle/toxic_comments'
df  = pd.read_csv("{}/original.csv".format(PWD), sep=',')

df['split'] = np.random.randn(df.shape[0], 1)

msk   = np.random.rand(len(df)) <= (1.0 - TEST_SIZE)
train = df[msk]
test  = df[~msk]

train.to_csv("{}/train.csv".format(PWD), sep=',')
test.to_csv("{}/test.csv".format(PWD), sep=',')
