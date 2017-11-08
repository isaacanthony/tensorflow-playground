import numpy as np
import tensorflow.contrib.keras as keras

dataset = np.loadtxt('dna/dna.csv', delimiter=',')

X = keras.backend.one_hot(dataset[:, 1:], 4)
Y = dataset[:, 0]

model = keras.models.Sequential([
  keras.layers.Dense(12, input_shape=X.shape[1:], activation='relu'),
  keras.layers.Dense(6, activation='relu'),
  keras.layers.Flatten(),
  keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

viz = keras.callbacks.TensorBoard(log_dir='logs',
                                  write_graph=True)

model.fit(X, Y, epochs=100, batch_size=10, callbacks=[viz])

model.evaluate(X, Y)

print 'Complete.'
