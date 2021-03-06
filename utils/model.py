import tensorflow as tf
import utils.config as config
import logging

class mlPerceptron:
  def model():
    LAYERS = [
            tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"),
            tf.keras.layers.Dense(config.layer1, activation="relu", name="hiddenLayer1"),
            tf.keras.layers.Dense(config.layer2, activation="relu", name="hiddenLayer2"),
            tf.keras.layers.Dense(10, activation="softmax", name="outputLayer")
    ]
    model_clf = tf.keras.models.Sequential(LAYERS)

    LOSS_FUNCTION = "sparse_categorical_crossentropy"
    OPTIMIZER = "SGD"
    METRICS = ["accuracy"]

    model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)
    logging.info(model_clf.summary())
    return model_clf
 