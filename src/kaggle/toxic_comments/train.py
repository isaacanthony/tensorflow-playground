import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow.contrib.keras as keras

# 1. Load data into memory.

PWD     = 'kaggle/toxic_comments'
dataset = pd.read_csv('{}/train.csv'.format(PWD), sep=',')

X = dataset['comment_text']
Y = dataset['toxic']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# 2. Preprocess text fields.

tokenizer = keras.preprocessing.text.Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X_train)

sequences = tokenizer.texts_to_sequences(X_train)
X_train   = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=300)

sequences = tokenizer.texts_to_sequences(X_test)
X_test    = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=300)

# 3. Build model.

model = keras.models.Sequential([
    keras.layers.Embedding(20000, 128, input_length=300),
    keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

viz = keras.callbacks.TensorBoard(log_dir='logs',
                                  write_graph=True)

# 4. Train model.

model.fit(X_train, Y_train, epochs=1, batch_size=32, callbacks=[viz])

# 5. Test model.

test_loss, test_acc = model.evaluate(X_test, Y_test)

print("\nTest accuracy:", test_acc)
print('Complete.')
