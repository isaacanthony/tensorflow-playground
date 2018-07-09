import tensorflow.contrib.keras as keras

PWD   = 'products'
MODEL = "{}/model.h5".format(PWD)

# Load model.
model = keras.models.load_model(MODEL)

# Stream test photos.
test_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    "{}/test".format(PWD),
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical')

# Test model.
results = model.evaluate_generator(test_generator)

print("loss: {}".format(results[0]))
print("accuracy: {}".format(results[1]))
