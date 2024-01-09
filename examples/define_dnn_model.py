#Define DNN model
# Define the TensorFlow model without any Horovod-specific parameters
def get_model():
    from tensorflow.keras import models
    from tensorflow.keras import layers

    model = models.Sequential()
    model.add(
        layers.Conv2D(32,
                      kernel_size=(3, 3),
                      activation='relu',
                      input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    return model
