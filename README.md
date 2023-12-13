# carPlateChar-recognition-wCNN
Car plate's characters recognition using convolutional neural network

## Running on Windows with GPU

1. https://www.tensorflow.org/install/pip#windows-native
2. Install Miniconda (used Anaconda in this case, but it's the same)
3. ``conda create --name tf python=3.9``
4. ``conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0``
5. ``pip install --upgrade pip``
6. ``pip install "tensorflow<2.11"``
7. Verify by running ``python check_gpu.py``
or by running
``python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"``