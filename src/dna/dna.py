import numpy as np
import tensorflow.contrib.keras as keras

dataset = np.loadtxt('dna/dna.csv', dtype=int, delimiter=',')

X = np.eye(4)[dataset[:, 1:]]
Y = dataset[:, 0]

print(X.shape)
print(Y.shape)

model = keras.models.Sequential([
  keras.layers.Conv1D(64, 20, input_shape=X.shape[1:]),
  keras.layers.Activation('relu'),
  keras.layers.MaxPooling1D(pool_size=4),
  keras.layers.Conv1D(32, 20),
  keras.layers.Activation('relu'),
  keras.layers.MaxPooling1D(pool_size=4),
  keras.layers.Flatten(),
  keras.layers.Dense(12, activation='relu'),
  keras.layers.Dense(1, activation='sigmoid')
])

print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

viz = keras.callbacks.TensorBoard(log_dir='logs',
                                  write_graph=True)

model.fit(X, Y, epochs=100, batch_size=10, callbacks=[viz])

model.evaluate(X, Y)

print('Complete.')
