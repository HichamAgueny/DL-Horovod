##prepare the dataset for training with the use the MNIST dataset
def get_dataset(rank=0, size=1):
    from tensorflow import keras
    from tensorflow.keras.datasets import mnist
    import numpy as np

    # Set a random seed for reproducibility
    np.random.seed(42)

    # Download MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data('MNIST-data-%d' % rank)
    #(x_train, y_train), (x_test, y_test) = mnist.load_data('MNIST-data-%d' % rank)
    # Pre-process data

    # Prepare dataset for distributed training
    x_train = x_train[rank::size]
    y_train = y_train[rank::size]
    x_test = x_test[rank::size]
    y_test = y_test[rank::size]

    # Reshape and Normalize data for model input
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)
