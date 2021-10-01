import tensorflow as tf

class mlPerceptron:
  def model():
    LAYERS = [
            tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"),
            tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
            tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
            tf.keras.layers.Dense(10, activation="softmax", name="outputLayer")
    ]
    model_clf = tf.keras.models.Sequential(LAYERS)

    LOSS_FUNCTION = "sparse_categorical_crossentropy"
    OPTIMIZER = "SGD"
    METRICS = ["accuracy"]

    model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)
    model_clf.summary()
    return model_clf
 