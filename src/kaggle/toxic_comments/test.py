import pandas as pd
import pickle
import tensorflow.contrib.keras as keras

# 1. Load data into memory.

PWD = 'kaggle/toxic_comments'
df  = pd.read_csv('{}/test.csv'.format(PWD), sep=',')

ID = df['id']
X  = df['comment_text'].fillna('')

# 2. Load model into memory.

with open("{}/tokenizer.pickle".format(PWD), 'rb') as f:
   tokenizer = pickle.load(f)

model = keras.models.load_model("{}/model.hdf5".format(PWD))

# 3. Preprocess text fields.

sequences = tokenizer.texts_to_sequences(X)
X         = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=300)

# 4. Output predictions.

results = model.predict(X)
ID = ID.apply(lambda id: [id])
results.insert(0, ID)
results = list(map(list, zip(*results)))

with open("{}/submission.csv".format(PWD), 'w') as f:
  f.write("id,toxic,severe_toxic,obscene,threat,insult,identity_hate\n")

  for i in range(len(results)):
    f.write(','.join(str(cell[0]) for cell in results[i]) + "\n")
