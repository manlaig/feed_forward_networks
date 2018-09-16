import tensorflow as tf
import matplotlib.pyplot as plot
import numpy as np
from tensorflow.keras.callbacks import TensorBoard

mnist = tf.keras.datasets.mnist

(image, answer), (i, t) = mnist.load_data()

image = tf.keras.utils.normalize(image, axis=1)
i = tf.keras.utils.normalize(i, axis=1)

"""
model = tf.keras.models.Sequential()
tensorboard = TensorBoard(log_dir='logs/')

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(np.array(image), np.array(answer), epochs=3, callbacks=[tensorboard])

model.save("mnist_digit_model.model")
"""

model = tf.keras.models.load_model("mnist_digit_model.model")

pred = model.predict([i])

for j in range(len(i)):
  plot.imshow(i[j], cmap='gray')
  plot.show()

  print("Predicted: " + str(np.argmax(pred[j])))
  