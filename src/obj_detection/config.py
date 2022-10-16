import torch

BATCH_SIZE = 4 # Increase / decrease according to GPU memory
RESIZE_TO = 512 # Resize the image for training and transforms
NUM_EPOCHS = 40 # Number of epochs to train for

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Training images and XML files directory
TRAIN_DIR = '../data/train'

# Validation images and XML files directory
VALID_DIR = '../data/test'

# Classes: 0 index is reserved for background, must contain background class.
CLASSES = [
    'background', 'cup', 'plate',
]
NUM_CLASSES = 3

# Whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# Location to save model and plots
OUT_DIR = '../outputs'
SAVE_PLOTS_EPOCH = 2 # Save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # Save model after these many epochs
