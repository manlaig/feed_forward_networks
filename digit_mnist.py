import tensorflow as tf
import matplotlib.pyplot as plot
import numpy as np

mnist = tf.keras.datasets.mnist

(image, answer), (i, t) = mnist.load_data()

image = tf.keras.utils.normalize(image, axis=1)

#model = tf.keras.models.load_model("mnist_digit_model.model")

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(np.array(image), np.array(answer), epochs=3)
model.save("mnist_digit_model.model")

pred = model.predict([i]);

for j in range(len(i)):
  plot.imshow(i[j])
  plot.show()

  print("Predicted: " + str(np.argmax(pred[j])))