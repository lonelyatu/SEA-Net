# SEA-Net
SEA-Net: Structure-Enhanced Attention Network for Limited-Angle CBCT Reconstruction of Clinical Projection Data

# Preparation
Download the pre-trained VGG16 weights (vgg_16.ckpt) at https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models.
Then put the vgg_16.ckpt into the "Code" folder.

# Training for SEA-Net
1. Enter the folder "Code" and run the tfrecordOp.py with your own training dataset path. Then the training data will be saved in tfrecord file.
2. After generating the training data, you can run the Main_SEA-Net.py to train your own model.

# Testing for SEA-Net
1. run "python Test.py" to test the files

# Environment

cuda 10.0

python 3.6.13

TensorFlow 1.15.4

Numpy 1.16.0

Scipy 1.2.1

