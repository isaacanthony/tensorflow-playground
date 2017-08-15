import tensorflow.contrib.keras as keras
import numpy

dataset = numpy.loadtxt('/src/iris.csv', delimiter=',')

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

model.fit(X, Y, epochs=100, batch_size=10)

model.evaluate(X, Y)

print 'Complete.'
