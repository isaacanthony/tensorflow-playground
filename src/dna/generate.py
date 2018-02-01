import random

PATH = 'dna/original.csv'

with open(PATH, 'w') as f:
    f.write("target,dna\n")

with open(PATH, 'a') as f:
    for i in range(0, 1000):
        y = random.sample(['0', '1'], 1)[0]
        x = []
        for j in range(0, 1024):
            x.append(random.sample(['A', 'C', 'G', 'T'], 1)[0])
        x = ''.join(x)
        f.write(','.join([y, x]) + "\n")
