import tensorflow as tf

model2 = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(2,100, 3)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])