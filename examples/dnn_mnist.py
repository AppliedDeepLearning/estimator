from estimator import Classifier, GradientDescent
import numpy as np
import tensorflow as tf


# Define the network architecture
def network(x, mode):
    x = tf.layers.Flatten()(x)
    x = tf.layers.Dense(units=50, activation=tf.nn.relu)(x)
    x = tf.layers.Dense(units=10)(x)
    return x

# Configure the model parameters
model = Classifier(network,
                   loss='sparse_softmax_cross_entropy',
                   optimizer=GradientDescent(0.001))

# Prepare data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.asarray(x_train, dtype=np.float32)
x_test = np.asarray(x_test, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.int32)
y_test = np.asarray(y_test, dtype=np.int32)

# Train the model using training data
model.train(x_train, y_train, num_epochs=1, batch_size=100)

# Evaluate the model performance on test or validation data
print(model.evaluate(x_test, y_test))

# Use the model to make predictions
print(list(model(x_test[0:10])))
