import os
import sys
import tensorflow.contrib.keras as keras
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

path = sys.argv[1]

img = keras.preprocessing.image.load_img(path, grayscale=False, target_size=(299,299))
img = keras.preprocessing.image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = keras.applications.inception_v3.preprocess_input(img)

model = keras.applications.inception_v3.InceptionV3(include_top=True,
                                                    weights='imagenet',
                                                    input_tensor=None,
                                                    input_shape=(299, 299, 3),
                                                    pooling=None,
                                                    classes=1000)

results = model.predict(img)
labels  = keras.applications.inception_v3.decode_predictions(results, top=3)[0]

for label in labels:
  print label[1]
