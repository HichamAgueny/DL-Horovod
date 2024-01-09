# Define a straing fct
from prepare_dataset import get_dataset
from define_dnn_model import get_model

def train(learning_rate,batch_size,epochs):
    # Import base libs
    import tempfile
    import os
    import shutil
    import atexit

    # Import tensorflow modules
    import tensorflow as tf
    from tensorflow import keras

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Prepare dataset
    (x_train, y_train), (x_test, y_test) = get_dataset()

    # Initialize model
    model = get_model()

    # Specify the optimizer (Adadelta in this example)
    optimizer = keras.optimizers.Adadelta(learning_rate=learning_rate)

    # Compile the model    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Model checkpoint location.
    #create temporary dir on the filesystem
    checkpoint_dir = tempfile.mkdtemp()
    #specify the file path for the model checkpoint within the temporary directory.
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.h5')
    #cleanup the temporary dir "ckpt_dir": it removes the dir and its content
    atexit.register(lambda: shutil.rmtree(checkpoint_dir))

    #save model checkpoints during training
    callbacks = tf.keras.callbacks.ModelCheckpoint(checkpoint_file,
                                                 monitor='val_loss',
                                                 mode='min',
                                                 save_best_only=True)

    # Train the model
    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              callbacks=callbacks,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test))

    # Return model bytes-like object
    with open(checkpoint_file, 'rb') as f:
        return f.read() #returns it as a bytes object.

#    return model

