# import the necessary packages
import os
import torch

# DEVICE AGNOSTIC
#### set device to 'cpu' or 'cuda' (GPU) based on availability
#### for model training and testing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MODEL HYPERPARAMETER
LR = 0.001      # learning rate for the model (weight and biases are updated on each iteration based on LR ) ~ alpha

IMAGE_SIZE = 512        # defines that the size of image is 512x512
CHANNELS = 3            # defines that image's color channel is 3
BATCH_SIZE = 16         # defines that 16 training examples are used in one iteration before
                        # updating the weight and biases of network

NUM_WORKERS = os.cpu_count()        # for parallel processing

EPOCHS = 200                 # number of complete passes through the entire training dataset
                             # i.e. model is trained on the whole dataset 10 times


# DATASET PATH
# CELEBFACE_ROOT = "./data/img_align_celeba"
CELEBFACE_ROOT = "./dataset_new/"


# OUTPUT DIRECTORY
# output_dir = "output"
# os.makedirs("output", exist_ok=True)

# # creates the training_progress directory inside the output directory
# training_progress_dir = os.path.join(output_dir, "training_progress")
# os.makedirs(training_progress_dir, exist_ok=True)


# # creates the model_weights directory inside the output directory
# # for storing autoencoder weights
# model_weights_dir = os.path.join(output_dir, "model_weights")
# os.makedirs(model_weights_dir, exist_ok=True)
