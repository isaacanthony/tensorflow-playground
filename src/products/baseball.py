import numpy as np
import tensorflow.contrib.keras as keras

PWD       = 'products'
MODEL     = "{}/model.h5".format(PWD)
THRESHOLD = 0.75

# Load model.
model = keras.models.load_model(MODEL)

# Stream test photos.
test_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    "{}/baseball".format(PWD),
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

# Test model.
results = model.predict_generator(test_generator)

count = 0.0
total = 0.0

for i in range(0, len(results)):
    predicted = np.argmax(results[i])
    value     = results[i][predicted]
    actual    = test_generator.classes[i]

    if value > THRESHOLD:
        if predicted == actual:
            count = count + 1
        total = total + 1

print("accuracy: {}".format(count / total))
print("coverage: {}".format(total / len(results)))
