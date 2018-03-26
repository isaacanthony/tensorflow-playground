import tensorflow.contrib.keras as keras

# 1. Import non-trainable inception model

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
y = keras.layers.Dense(1, activation='sigmoid')(y)

model = keras.models.Model(inputs=base.input, outputs=y)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 3. Stream photos.

train_datagen = keras.preprocessing.image.ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    'pinkeye/downloads',
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=3)
