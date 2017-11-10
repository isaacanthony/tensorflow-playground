import os
import numpy as np
import tensorflow.contrib.keras as keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. Stream photos

dataset = np.loadtxt('product_categories.csv',
                     delimiter=',',
                     skiprows=1)

X = []
Y = keras.utils.to_categorical(dataset[:, 1], num_classes=6)

for row in dataset:
  path = "photos/%s.jpg" % int(row[0])
  img = keras.preprocessing.image.load_img(path, grayscale=False, target_size=(299,299))
  img = keras.preprocessing.image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  img = keras.applications.inception_v3.preprocess_input(img)
  X.append(img[0])

X = np.array(X)
Y = np.array(Y)

# 2. Import non-trainable inception model

base = keras.applications.inception_v3.InceptionV3(include_top=True,
                                                   weights='imagenet',
                                                   input_tensor=None,
                                                   input_shape=(299, 299, 3),
                                                   pooling=None,
                                                   classes=1000)

for layer in base.layers:
  layer.trainable = False

# 3. Add extra layers on top

y = base.output
y = keras.layers.Dense(128, input_dim=1000, activation='relu')(y)
y = keras.layers.Dropout(0.5)(y)
y = keras.layers.Dense(32, activation='relu')(y)
y = keras.layers.Dense(6, activation='sigmoid')(y)

model = keras.models.Model(inputs=base.input, outputs=y)

# 4. Train model

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

viz = keras.callbacks.TensorBoard(log_dir='logs',
                                  write_graph=True)

model.fit(X, Y, epochs=100, batch_size=10, callbacks=[viz])

model.evaluate(X, Y)

# 5. Export weights

model.save_weights('product_category_weights.h5')
