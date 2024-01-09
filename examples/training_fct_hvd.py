# Define a straing fct
from prepare_dataset import get_dataset
from define_dnn_model import get_model

def train_hvd(learning_rate,batch_size,epochs):
    # Import base libs
    import tempfile
    import os
    import shutil
    import atexit

    # Import tensorflow modules
    import tensorflow as tf
    from tensorflow import keras
    import horovod.tensorflow.keras as hvd

    # Initialize Horovod
#    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
#    gpus = tf.config.experimental.list_physical_devices('GPU')
#    for gpu in gpus:
#        tf.config.experimental.set_memory_growth(gpu, True)
#        print(f"GPU name: {gpu.name}")
#    if gpus:
#        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()],'GPU')
#        print(f"Number of GPUs available: {len(gpus)}")
#    else:
#        print("No GPUs available.")        
                
    # Prepare dataset with the use of Horovod rank and size
    #The 2 arguments hvd.rank() are hvd.size() passed to the fct get_dataset
    # hvd.rank returns the ID-rank of the current process
    # hvd.size returns the total nbr of processes
    # the data is partioned according to the nbr of processes
    (x_train, y_train), (x_test, y_test) = get_dataset(hvd.rank(), hvd.size())

    # Create TensorFlow datasets
    #train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    #test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    #print("--rank,len,size", hvd.rank(),len(x_train),x_train.size,len(x_test),x_test.size)


    # Initialize DNN model
    model = get_model()

    # Specify the optimizer:
    # Scale the learning rate with the number of GPUs (total nbr of processes)
    #When partioning the data according to the nbr of processes, the batch size on each
    #process is the same as it would be in a single process (non-distributed training)
    #Here each worker processes a smaller subset of the data, but combining all processes
    #results in a larger batch size. 
    #Ex. Let's consider a batch size of 64 on each process and that hvd.size()=4. 
    #Each process processes its unique subset of data, but collectively
    #the effective batch size is 64x4=256 data points (larger than a single-process)
    #Thus, the learnin rate should be scaled to account for the increased eefective batch size.

    optimizer = keras.optimizers.Adadelta(learning_rate=learning_rate *
                                          hvd.size())

    # Use the Horovod Distributed Optimizer
    #The primary purpose of this wrapper is to handle the communication and synchronization 
    #of gradients during distributed training. 
    #Horovod's DistributedOptimizer ensures that gradient updates from different workers are correctly aggregated and applied to the model. 
    optimizer = hvd.DistributedOptimizer(optimizer)

    # Compile the model    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Create a callback to broadcast the initial variable states from rank 0 to all other processes.
    # This is required to ensure consistent initialization of all workers when training is started with random weights or restored from a checkpoint.
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    ]

    # Model checkpoint location.
    #create temporary dir on the filesystem
    checkpoint_dir = tempfile.mkdtemp()
    #specify the file path for the model checkpoint within the temporary directory.
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.h5')
    #cleanup the temporary dir "ckpt_dir": it removes the dir and its content
    atexit.register(lambda: shutil.rmtree(checkpoint_dir))

    # Save checkpoints during training: only on worker 0 to prevent conflicts between workers
    if hvd.rank() == 0:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(checkpoint_file,
                                            monitor='val_loss',
                                            mode='min',
                                            save_best_only=True))

    # Train the model
    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              callbacks=callbacks,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test))

    # Return model bytes-like object only worker 0
    if hvd.rank() == 0:
        with open(checkpoint_file, 'rb') as f:
            return f.read() #returns it as a bytes object.

    #return model

