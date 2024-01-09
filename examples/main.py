# Specify training parameters
num_proc = 1  # equal to numExecutors
batch_size = 128
epochs = 1
lr_single_node = 0.1  # learning rate for single node code
#num_classes=10

import os
from training_fct import train 
from prepare_dataset import get_dataset

# Run the training process on the driver
print('Running training ....')
best_model_bytes = train(learning_rate=lr_single_node, batch_size=batch_size, epochs=epochs)


# Save checkpoints

# Specify the path to the directory where checkpoints are stored
checkpoint_dir = './data_train'

# Create the checkpoint directory if it doesn't exist
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

print(f"Checkpoint Directory: {checkpoint_dir}")

# Specify the file path for the model checkpoint within the directory
checkpoint_file = os.path.join(checkpoint_dir, 'mnist-ckpt.h5')

# Now you can open the checkpoint file for writing binary data
with open(checkpoint_file, 'wb') as f:
    f.write(best_model_bytes)

# Save the model to a file
# designed for serializing Keras models to a file, and it will correctly save the model in a format that can later be loaded and used for predictions or further training.
#model.save(checkpoint_file)

# Evaluate the trained model
import tensorflow as tf

# Load a saved Keras model from a file:
# it reads the saved model structure, architecture, and weights from the file specified by checkpoint_file.
# the returned loaded model is stored in 'model_gpu'
model_gpu = tf.keras.models.load_model(checkpoint_file)

_, (x_test, y_test) = get_dataset()
loss, accuracy = model_gpu.evaluate(x_test, y_test, batch_size=batch_size)
print("loss:", loss)
print("accuracy:", accuracy)

