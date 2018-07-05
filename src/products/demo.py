import sys
import numpy as np
import tensorflow.contrib.keras as keras

PWD   = 'products'
MODEL = "{}/model.h5".format(PWD)

# Import photo

path = sys.argv[1]

img = keras.preprocessing.image.load_img(path, grayscale=False, target_size=(299,299))
img = keras.preprocessing.image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = keras.applications.inception_v3.preprocess_input(img)

# Import model

model = keras.models.load_model(MODEL)

# Print prediction

results = model.predict(img)

print(results)
