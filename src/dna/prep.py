import pandas as pd
import numpy as np

TEST_SIZE = 0.2

df = pd.read_csv('dna/original.csv', sep=',')

df['split'] = np.random.randn(df.shape[0], 1)

msk   = np.random.rand(len(df)) <= (1.0 - TEST_SIZE)
train = df[msk]
test  = df[~msk]

train.to_csv('dna/train.csv', sep=',')
test.to_csv('dna/test.csv', sep=',')
