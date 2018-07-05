import os.path
import tensorflow.contrib.keras as keras
import time

PWD   = 'products'
MODEL = "{}/model.h5".format(PWD)
LOGS  = "logs/{}:{}".format(time.localtime().tm_hour, time.localtime().tm_min)

if os.path.isfile(MODEL):
    # Load model if already exists.
    model = keras.models.load_model(MODEL)

else:
    # Import non-trainable inception model.
    base = keras.applications.inception_v3.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=(299, 299, 3),
        pooling='avg')

    for layer in base.layers:
        layer.trainable = False

    # Add extra layers on top.
    y = base.output
    y = keras.layers.Dense(32, activation='relu')(y)
    y = keras.layers.Dense(6, activation='softmax')(y)

    # Construct model.
    model = keras.models.Model(inputs=base.input, outputs=y)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Stream photos.
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    channel_shift_range=0.1,
    horizontal_flip=True)

# Train model.
train_generator = train_datagen.flow_from_directory(
    "{}/downloads".format(PWD),
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical')

viz  = keras.callbacks.TensorBoard(log_dir=LOGS, write_graph=True)
save = keras.callbacks.ModelCheckpoint(MODEL, period=1)

model.fit_generator(
    train_generator,
    steps_per_epoch=150,
    epochs=10,
    callbacks=[viz, save])
