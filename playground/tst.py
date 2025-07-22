import tensorflow as tf

# Check if TensorFlow can access a GPU
if tf.config.list_physical_devices('GPU'):
    print("CUDA is available. You are using GPU!")
else:
    print("CUDA is not available. Using CPU instead.")
