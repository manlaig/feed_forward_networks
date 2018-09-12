import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

(img, ans), (img_test, ans_test) = fashion_mnist.load_data()

img = tf.keras.utils.normalize(img, axis=1)
"""
plt.imshow(img_test[0])
plt.show()
print("Correct: " + str(ans_test[0]))

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
             metrics=["accuracy"])

model.fit(img, ans, epochs=3)

model.save("fashion_mnist.model")
"""
model = tf.keras.models.load_model("fashion_mnist.model")

i = model.predict(img_test)
print("Predicted: " + str(i[2]))
print("Correct: " + str(ans_test[2]))
plt.imshow(img_test[2])
plt.show()
