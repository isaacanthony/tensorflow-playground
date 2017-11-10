import tensorflow.contrib.keras as keras

# 1. Import data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# 2. Preprocess data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images  = test_images.reshape((10000, 28 * 28))
test_images  = test_images.astype('float32') / 255

train_labels = keras.utils.to_categorical(train_labels)
test_labels  = keras.utils.to_categorical(test_labels)

# 3. Build model
model = keras.models.Sequential([
    keras.layers.Dense(512, input_shape=(28 * 28,), activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

viz = keras.callbacks.TensorBoard(log_dir='logs',
                                  write_graph=True)

# 4. Train network
model.fit(train_images,
          train_labels,
          epochs=5,
          batch_size=128,
          callbacks=[viz])

# 5. Test model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("\nTest accuracy:", test_acc)
print('Complete.')
