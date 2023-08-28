# SEA-Net
SEA-Net: Structure-Enhanced Attention Network for Limited-Angle CBCT Reconstruction of Clinical Projection Data

# Preparation
Download the pre-trained VGG16 weights (vgg_16.ckpt) at https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models.
Then put the vgg_16.ckpt into the "Code" folder.

# Training for SEA-Net
1. Enter the folder "Code" and run "python TFRecordOp.py" and the files in the "TFRecordsFile" will be saved as tfrecord files

2. run "python main_PriorNet.py" and the model will be saved every epoch

# Testing for PRIOR-Net
1. run "python Test_PriorNet.py" to test the files

# Testing for PRIOR
1. Enter the folder "PRIOR" and run "python main_PRIOR.py". Before run the PRIOR, you should first train the PRIOR-Net or directly use the trained model we provided in the folder "PriorNet" 

# Environment

cuda 10.0

python 3.6.13

TensorFlow 1.15.4

Numpy 1.16.0

Scipy 1.2.1

tigre (https://github.com/CERN/TIGRE)
