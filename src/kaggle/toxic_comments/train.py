import os.path
import pandas as pd
import pickle
import tensorflow.contrib.keras as keras

# 1. Load data into memory.

PWD   = 'kaggle/toxic_comments'
train = pd.read_csv("{}/train.csv".format(PWD), sep=',')
test  = pd.read_csv("{}/test.csv".format(PWD), sep=',')

X_train  = train['comment_text']
Y1_train = train['toxic']
Y2_train = train['severe_toxic']
Y3_train = train['obscene']
Y4_train = train['threat']
Y5_train = train['insult']
Y6_train = train['identity_hate']

X_test  = test['comment_text']
Y1_test = test['toxic']
Y2_test = test['severe_toxic']
Y3_test = test['obscene']
Y4_test = test['threat']
Y5_test = test['insult']
Y6_test = test['identity_hate']

# 2. Preprocess text fields.

path = "{}/tokenizer.pickle".format(PWD)
if os.path.isfile(path):
  with open(path, 'rb') as f:
    tokenizer = pickle.load(f)
else:
  tokenizer = keras.preprocessing.text.Tokenizer(num_words=20000, char_level=True)
  tokenizer.fit_on_texts(X_train)
  with open(path, 'wb') as f:
     pickle.dump(tokenizer, f)

sequences = tokenizer.texts_to_sequences(X_train)
X_train   = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=512)

sequences = tokenizer.texts_to_sequences(X_test)
X_test    = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=512)

# 3. Build model.

ins = keras.layers.Input(shape=(512,), dtype='int32')
y   = keras.layers.Embedding(20000, 128)(ins)
y   = keras.layers.Dropout(0.2)(y)
y   = keras.layers.Conv1D(32, 5, activation='relu')(y)
y   = keras.layers.MaxPooling1D(pool_size=4)(y)
y   = keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)(y)
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
