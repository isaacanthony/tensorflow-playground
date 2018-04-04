import os.path
import tensorflow.contrib.keras as keras

# 1. Import non-trainable inception model

PWD = 'pinkeye'

base = keras.applications.inception_v3.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(299, 299, 3),
    pooling='max',
    classes=1000)

for layer in base.layers:
    layer.trainable = False

# 2. Add extra layers on top

y = base.output
y = keras.layers.Dense(64, input_dim=1000, activation='relu')(y)
y = keras.layers.Dense(32, activation='relu')(y)
y = keras.layers.Dense(3, activation='sigmoid')(y)

path = "{}/model.hdf5".format(PWD)
if os.path.isfile(path):
    model = keras.models.load_model(path)
else:
    model = keras.models.Model(inputs=base.input, outputs=y)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 3. Stream photos.

train_datagen = keras.preprocessing.image.ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    "{}/downloads".format(PWD),
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical')

viz  = keras.callbacks.TensorBoard(log_dir='logs', write_graph=True)
save = keras.callbacks.ModelCheckpoint(path, period=1)

model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=3,
    callbacks=[viz, save])
