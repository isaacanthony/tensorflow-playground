import pandas as pd
import tensorflow.contrib.keras as keras

HOME_DIR = 'kaggle/toxic_comments'
dataset  = pd.read_csv("{}/train.csv".format(HOME_DIR), sep=',')

X = dataset['comment_text']
Y = dataset['toxic']

tokenizer = keras.preprocessing.text.Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
X = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=300)

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

model.fit(X, Y, epochs=12, batch_size=64, callbacks=[viz])

model.evaluate(X, Y)

print("\nComplete.")
