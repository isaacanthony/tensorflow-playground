import os
import numpy as np
import tensorflow.contrib.keras as keras

# 1. Load photos

PWD     = 'kaggle/imaterialist'
dataset = np.loadtxt("{}/validation.csv".format(PWD), delimiter=',', skiprows=1)

X = []
Y = keras.utils.to_categorical(dataset[:, 1].astype(int), num_classes=129)

for row in dataset:
  path = "{}/validation/{}".format(PWD, row[0].astype(int))
  img = keras.preprocessing.image.load_img(path, grayscale=False, target_size=(299,299))
  img = keras.preprocessing.image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  img = keras.applications.inception_v3.preprocess_input(img)
  X.append(img[0])

X = np.array(X)
Y = np.array(Y)

print(X.shape)
print(Y.shape)

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
y = keras.layers.Dense(129, input_dim=1000, activation='relu')(y)

# 4. Load or create model

path = "{}/model.hdf5".format(PWD)
if os.path.isfile(path):
  model = keras.models.load_model(path)
else:
  model = keras.models.Model(inputs=base.input, outputs=y)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. Train model

viz  = keras.callbacks.TensorBoard(log_dir='logs', write_graph=True)
save = keras.callbacks.ModelCheckpoint(path, period=1)

model.fit(X, Y, epochs=6, batch_size=32, callbacks=[viz, save])
model.evaluate(X, Y)
