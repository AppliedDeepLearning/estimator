from estimator import Model, TRAIN, cli
import numpy as np
import tensorflow as tf

args = cli(learning_rate=0.001,
           epochs=30,
           batch_size=128,
           model_dir=None)


# Define the network architecture
def network(x, mode):
    x = tf.reshape(x, [-1, 28, 28, 1])
    x = tf.layers.Conv2D(filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)(x)
    x = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(x)
    x = tf.layers.Conv2D(filters=64, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)(x)
    x = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(x)
    x = tf.layers.Flatten()(x)
    x = tf.layers.Dense(units=1024, activation=tf.nn.relu)(x)
    x = tf.layers.Dropout(rate=0.4)(x, training=mode == TRAIN)
    x = tf.layers.Dense(units=10)(x)
    return x

# Configure the model parameters
model = Model(network,
              loss='sparse_softmax_cross_entropy',
              optimizer=('GradientDescent', args.learning_rate),
              metrics=['accuracy'],
              model_dir=args.model_dir)

# Prepare data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.asarray(x_train, dtype=np.float32)
x_test = np.asarray(x_test, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.int32)
y_test = np.asarray(y_test, dtype=np.int32)

# Train the model using training data
model.train(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size)

# Evaluate the model performance on test or validation data
print(model.evaluate(x_test, y_test))

# Use the model to make predictions
print(list(model(x_test[0:10])))
