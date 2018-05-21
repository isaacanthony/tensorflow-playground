import pandas as pd

# 1. Import data
PWD    = 'horses'
horses = pd.read_csv("{}/horse.csv".format(PWD))
races  = pd.read_csv("{}/race.csv".format(PWD))
df     = horses.join(races,
                     on='race_id',
                     how='left',
                     lsuffix='_left',
                     rsuffix='_right')

# 2. Clean data
print('columns')
print(df.columns.values)
print()

print('finishing_position')
print(df.finishing_position.unique())
df.fillna('0', inplace=True)
df.finishing_position.replace('.*[A-Z].*', '0', regex=True, inplace=True)
df.finishing_position = df.finishing_position.astype('int32').astype('category')
print(df.finishing_position.unique())
print()

print('horse_number')
print(df.horse_number.unique())
df.horse_number.replace('0', 0, inplace=True)
df.horse_number = df.horse_number.astype('int32').astype('category')
print(df.horse_number.unique())
print()

print('horse_name')
print(df.horse_name.unique())
df.horse_name = df.horse_name.str.lower().astype('category')
print(df.horse_name.unique())
print()

print('jockey')
print(df.jockey.unique())
df.jockey = df.jockey.str.lower().astype('category')
print(df.jockey.unique())
print()

print('trainer')
print(df.trainer.unique())
df.trainer = df.trainer.str.lower().astype('category')
print(df.trainer.unique())
print()

print('actual_weight')
print(df.actual_weight.unique())
df.actual_weight.replace('-', '0', inplace=True)
df.actual_weight = df.actual_weight.astype('float32')
actual_weight_max = df.actual_weight.max()
print('max: ', actual_weight_max)
df.actual_weight /= actual_weight_max
print(df.actual_weight.unique())
print()

print('declared_horse_weight')
print(df.declared_horse_weight.unique())
df.declared_horse_weight.replace('-', '0', inplace=True)
df.declared_horse_weight = df.declared_horse_weight.astype('float32')
declared_horse_weight_max = df.declared_horse_weight.max()
print('max: ', declared_horse_weight_max)
df.declared_horse_weight /= declared_horse_weight_max
print(df.declared_horse_weight.unique())
print()

print('draw')
print(df.draw.unique())
df.draw.replace('---', '0', inplace=True)
df.draw = df.draw.astype('int32').astype('category')
print(df.draw.unique())
print()

print('race_date')
print('race_course')
print('race_number')
print('race_class')
print('race_distance')
print('track_condition')
print('race_name')
print('track')

# 3. Save data
df.to_csv("{}/cleaned.csv".format(PWD), index=False)
