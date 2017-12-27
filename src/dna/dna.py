import numpy as np
import tensorflow.contrib.keras as keras

train = np.loadtxt('dna/train.csv', dtype=int, delimiter=',')
test  = np.loadtxt('dna/test.csv', dtype=int, delimiter=',')

X_train = np.eye(4)[train[:, 1:]]
Y_train = train[:, 0]

X_test = np.eye(4)[test[:, 1:]]
Y_test = test[:, 0]

print(X_train.shape)
print(Y_train.shape)

model = keras.models.Sequential([
  keras.layers.Conv1D(64, 20, input_shape=X_train.shape[1:]),
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

model.fit(X_train, Y_train, epochs=12, batch_size=10, callbacks=[viz])

test_loss, test_acc = model.evaluate(X_test, Y_test)

print("\nTest accuracy:", test_acc)
print('Complete.')
