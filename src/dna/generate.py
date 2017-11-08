import random

for i in range(0, 1000):
  y = random.sample(['0', '1'], 1)[0]
  x = []
  for j in range(0, 100):
    x.append(random.sample(['A', 'C', 'G', 'T'], 1)[0])
  x = ''.join(x)
  print ','.join([y, x])
