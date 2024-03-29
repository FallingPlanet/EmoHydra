# Import necessary modules from MMSA
from MMSA import MMSA_run

# Load the CMU-MOSEI dataset
dataset = CMUMOSEI()

# Access emotion labels from the dataset
emotion_labels = dataset.emotion_labels

# Save the dataset to your system
save_path = "path/to/save/dataset"
dataset.save(save_path)