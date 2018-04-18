import tensorflow as tf

from .models import Classifier, Regressor, Model, spec
from .utils import dataset, cli

# Optimizers
GradientDescent = tf.train.GradientDescentOptimizer
Adadelta = tf.train.AdadeltaOptimizer
Adagrad = tf.train.AdagradOptimizer
AdagradDA = tf.train.AdagradDAOptimizer
Momentum = tf.train.MomentumOptimizer
Adam = tf.train.AdamOptimizer
Ftrl = tf.train.FtrlOptimizer
ProximalGradientDescent = tf.train.ProximalGradientDescentOptimizer
ProximalAdagrad = tf.train.ProximalAdagradOptimizer
RMSProp = tf.train.RMSPropOptimizer

# Modes
TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL
PREDICT = tf.estimator.ModeKeys.PREDICT
