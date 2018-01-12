import numpy as np
import pandas as pd
import tensorflow.contrib.keras as keras

# 1. Import data.

PWD = 'kaggle/iceberg'
df  = pd.read_json("{}/train.json".format(PWD))

X0 = pd.DataFrame(df['inc_angle'])
X1 = pd.DataFrame(df['band_1'].values.tolist())
X2 = pd.DataFrame(df['band_2'].values.tolist())
Y  = pd.DataFrame(df['is_iceberg'])

# 2. Preprocess data.

X0 = X0.replace('na', np.nan).astype('float32') # .fillna(X0.values.mean())
X1 = X1.values.reshape(X1.shape[0], 75, 75, 1) / 35.0
X2 = X2.values.reshape(X1.shape[0], 75, 75, 1) / 21.0

# 3. Build model.

in0 = keras.layers.Input(shape=X0.shape[1:])

in1 = keras.layers.Input(shape=(75, 75, 1))
y1  = keras.layers.Conv2D(32, (5, 5))(in1)
y1  = keras.layers.Activation('relu')(y1)
y1  = keras.layers.MaxPooling2D(pool_size=(5, 5))(y1)
y1  = keras.layers.Conv2D(32, (5, 5))(y1)
y1  = keras.layers.Activation('relu')(y1)
y1  = keras.layers.MaxPooling2D(pool_size=(5, 5))(y1)

in2 = keras.layers.Input(shape=(75, 75, 1))
y2  = keras.layers.Conv2D(32, (5, 5))(in2)
y2  = keras.layers.Activation('relu')(y2)
y2  = keras.layers.MaxPooling2D(pool_size=(5, 5))(y2)
y2  = keras.layers.Conv2D(32, (5, 5))(y2)
y2  = keras.layers.Activation('relu')(y2)
y2  = keras.layers.MaxPooling2D(pool_size=(5, 5))(y2)

y = keras.layers.concatenate([y1, y2])
y = keras.layers.Flatten()(y)
y = keras.layers.Dense(32, activation='relu')(y)
y = keras.layers.Dense(16, activation='relu')(y)
y = keras.layers.Dense(1, activation='sigmoid')(y)

model = keras.models.Model(inputs=[in1, in2], outputs=y)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

viz = keras.callbacks.TensorBoard(log_dir='logs', write_graph=True)

# 4. Train model.

model.fit([X1, X2], Y, epochs=1, batch_size=32, callbacks=[viz])
