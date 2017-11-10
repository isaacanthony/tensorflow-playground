import numpy as np
import tensorflow.contrib.keras as keras

dataset = np.loadtxt('iris.csv', delimiter=',')

X = dataset[:, 0:4]
Y = keras.utils.to_categorical(dataset[:, 4], num_classes=3)

model = keras.models.Sequential([
    keras.layers.Dense(12, input_dim=4, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(3, activation='sigmoid')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

viz = keras.callbacks.TensorBoard(log_dir='logs',
                                  write_graph=True)

model.fit(X, Y, epochs=100, batch_size=10, callbacks=[viz])

model.evaluate(X, Y)

print("\nComplete.")
