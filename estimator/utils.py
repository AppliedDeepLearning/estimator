import tensorflow as tf
import logging


def call_fn(fn, *arguments, **keywords):
    fn_keywords = fn.__code__.co_varnames
    kwargs = {}
    for keyword, value in keywords.items():
        if keyword in fn_keywords:
            kwargs[keyword] = value
    return fn(*arguments, **kwargs)


def to_dense(x):
    return tf.argmax(x, axis=1) if len(x.shape) > 1 and x.shape[1] > 1 else x

to_categorical = tf.keras.utils.to_categorical

logger = logging.getLogger('estimator')
logger.setLevel(logging.INFO)
