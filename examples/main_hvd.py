# Specify training parameters
batch_size = 256
epochs = 1
lr_single_node = 0.1  # learning rate for single node code
#num_classes=10

import os
from training_fct_hvd import train_hvd 
from prepare_dataset import get_dataset
import tensorflow as tf
import horovod.tensorflow.keras as hvd
import numpy as np

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU name: {gpu.name}")
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()],'GPU')
    print(f"Number of GPUs available: {len(gpus)}")
else:
    print("No GPUs available.")

if hvd.rank() == 0:
    print('Running Distributed training with horovod ....')

print('Hello, rank = %d, local_rank = %d, size = %d, local_size = %d' % (hvd.rank(), hvd.local_rank(), hvd.size(), hvd.local_size()))

# Run distributed training
best_model_bytes = train_hvd(learning_rate=lr_single_node, batch_size=batch_size, epochs=epochs)

print("--batch_size", hvd.rank(), batch_size)
# Specify the path to the directory where checkpoints are stored
checkpoint_dir = './data_train'

# Create the checkpoint directory if it doesn't exist
if hvd.rank() == 0:
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    print(f"Checkpoint Directory: {checkpoint_dir}")

# Specify the file path for the model checkpoint within the directory
checkpoint_file = os.path.join(checkpoint_dir, 'mnist-ckpt.h5')

# Save checkpoints only on worker 0
if hvd.rank() == 0:
    callbacks = tf.keras.callbacks.ModelCheckpoint(checkpoint_file,
                                            monitor='val_loss',
                                            mode='min',
                                            save_best_only=True)

# Now you can open the checkpoint file for writing binary data
if hvd.rank() == 0:
    with open(checkpoint_file, 'wb') as f:
        f.write(best_model_bytes)

# Save the model to a file
# designed for serializing Keras models to a file, and it will correctly save the model in a format that can later be loaded and used for predictions or further training.
#model.save(checkpoint_file)

# Evaluate the trained model
    print('--Evaluate the trained model ...')

# Load a saved Keras model from a file:
# it reads the saved model structure, architecture, and weights from the file specified by checkpoint_file.
# the returned loaded model is stored in 'model_hvd'
    model_hvd = tf.keras.models.load_model(checkpoint_file)

    _, (x_test, y_test) = get_dataset()
    print("--x_test:", hvd.rank(),len(x_test), len(y_test))
# Create TensorFlow datasets
#test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    loss, accuracy = model_hvd.evaluate(x_test, y_test, batch_size=batch_size)
    print("loss:", loss)
    print("accuracy:", accuracy)

    # Use rint() to round the predicted values to the nearest integer
    #preds = np.rint(model_hvd.predict(x_test[0:9]))
    #print(preds)

# Aggregate evaluation metrics across all workers using Horovod
#avg_loss = hvd.allreduce(loss, average=True)
#avg_accuracy = hvd.allreduce(accuracy, average=True)

# Print or log aggregated metrics on rank 0
#if hvd.rank() == 0:
#    print("Aggregated loss:", avg_loss)
#    print("Aggregated accuracy:", avg_accuracy)
