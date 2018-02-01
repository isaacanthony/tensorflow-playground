import pandas as pd
import tensorflow.contrib.keras as keras

train = pd.read_csv('dna/train.csv', sep=',')
test  = pd.read_csv('dna/test.csv', sep=',')

X_train = train['dna']
Y_train = train['target']

X_test = test['dna']
Y_test = test['target']

tokenizer = keras.preprocessing.text.Tokenizer(num_words=4, char_level=True)
tokenizer.fit_on_texts(X_train)

sequences = tokenizer.texts_to_sequences(X_train)
X_train   = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=1024)

sequences = tokenizer.texts_to_sequences(X_test)
X_test    = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=1024)

model = keras.models.Sequential([
  keras.layers.Embedding(4, 4, input_shape=X_train.shape[1:]),
  keras.layers.Conv1D(64, 20, activation='relu'),
  keras.layers.MaxPooling1D(pool_size=4),
  keras.layers.Conv1D(32, 20, activation='relu'),
  keras.layers.MaxPooling1D(pool_size=4),
  keras.layers.Flatten(),
  keras.layers.Dense(12, activation='relu'),
  keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

viz = keras.callbacks.TensorBoard(log_dir='logs', write_graph=True)

model.fit(X_train, Y_train, epochs=12, batch_size=10, callbacks=[viz])

test_loss, test_acc = model.evaluate(X_test, Y_test)

print("\nTest accuracy:", test_acc)
print('Complete.')
