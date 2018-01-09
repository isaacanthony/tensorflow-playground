import os.path
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import tensorflow.contrib.keras as keras

# 1. Load data into memory.

PWD     = 'kaggle/toxic_comments'
dataset = pd.read_csv("{}/train.csv".format(PWD), sep=',')

X  = dataset['comment_text']
Y1 = dataset['toxic']
Y2 = dataset['severe_toxic']
Y3 = dataset['obscene']
Y4 = dataset['threat']
Y5 = dataset['insult']
Y6 = dataset['identity_hate']

split    = train_test_split(X, Y1, Y2, Y3, Y4, Y5, Y6, test_size=0.2)
X_train  = split[0]
X_test   = split[1]
Y1_train = split[2]
Y1_test  = split[3]
Y2_train = split[4]
Y2_test  = split[5]
Y3_train = split[6]
Y3_test  = split[7]
Y4_train = split[8]
Y4_test  = split[9]
Y5_train = split[10]
Y5_test  = split[11]
Y6_train = split[12]
Y6_test  = split[13]

# 2. Preprocess text fields.

tokenizer = keras.preprocessing.text.Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X_train)

with open("{}/tokenizer.pickle".format(PWD), 'wb') as f:
   pickle.dump(tokenizer, f)

sequences = tokenizer.texts_to_sequences(X_train)
X_train   = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=300)

sequences = tokenizer.texts_to_sequences(X_test)
X_test    = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=300)

# 3. Build model.

ins = keras.layers.Input(shape=(300,), dtype='int32')
y   = keras.layers.Embedding(20000, 128)(ins)
y   = keras.layers.Dropout(0.2)(y)
y   = keras.layers.Conv1D(64, 5, activation='relu')(y)
y   = keras.layers.MaxPooling1D(pool_size=4)(y)
y   = keras.layers.LSTM(128)(y)
y1  = keras.layers.Dense(1, activation='sigmoid')(y)
y2  = keras.layers.Dense(1, activation='sigmoid')(y)
y3  = keras.layers.Dense(1, activation='sigmoid')(y)
y4  = keras.layers.Dense(1, activation='sigmoid')(y)
y5  = keras.layers.Dense(1, activation='sigmoid')(y)
y6  = keras.layers.Dense(1, activation='sigmoid')(y)

path = "{}/model.hdf5".format(PWD)
if os.path.isfile(path):
  model = keras.models.load_model(path)
else:
  model = keras.models.Model(inputs=ins, outputs=[y1, y2, y3, y4, y5, y6])
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

viz  = keras.callbacks.TensorBoard(log_dir='logs', write_graph=True)
save = keras.callbacks.ModelCheckpoint(path, period=1)

# 4. Train model.

model.fit(X_train,
          [Y1_train, Y2_train, Y3_train, Y4_train, Y5_train, Y6_train],
          epochs=3,
          batch_size=32,
          callbacks=[viz, save])

# 5. Test model.

results = model.evaluate(X_test,
                         [Y1_test, Y2_test, Y3_test, Y4_test, Y5_test, Y6_test])

print("\nLoss:", results[0])
print('Accuracy:', results[-6:])
print('Complete.')
