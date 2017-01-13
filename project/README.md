Implementation of DropConnect paper

Project completed using E6040 AMI on g2.2xlarge instance.  Prior to running code, type in terminal:

source activate theano
pip install --upgrade theano

NOTE FOR NORB DATASET:

The NORB dataset is very large.  Prior to running the code with the NORB dataset, it may 
be necessary to restart the computer to allow room on the GPU memory to hold the images.

Each dataset has its own ipython notebook as well as its own testfile which contains the test functions for the networks.

Project_utils containes the functions for downloading and loading the data.

Projecy_nn containes the funtions for layers, MLP, DropConnect, Convlayers, pooling, lolcally connected layers.
