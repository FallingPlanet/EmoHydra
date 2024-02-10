import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import glob
import json

class MultimodalMELDDataset(Dataset):
    def __init__(self, base_dir, label_mapping, output_dir=None):
        self.base_dir = base_dir
        self.label_mapping = label_mapping
        self.output_dir = output_dir  # Optional output directory for saving processed data
        self.sample_info = self._gather_samples_info()

    def _gather_samples_info(self):
        """
        Gathers information about samples by walking through the text data directory structure.
        Constructs paths for vision and audio data based on the text data paths.
        """
        samples = []
        # Starting point is the text data directory
        text_base_dir = os.path.join(self.base_dir, "MELD_Text")
        
        # Iterate over each emotion directory within the text data directory
        for emotion in os.listdir(text_base_dir):
            emotion_dir = os.path.join(text_base_dir, emotion)
            if not os.path.isdir(emotion_dir):
                continue  # Skip if not a directory

            # Iterate over dialogue directories within each emotion directory
            for dialogue in os.listdir(emotion_dir):
                dialogue_dir = os.path.join(emotion_dir, dialogue)
                if not os.path.isdir(dialogue_dir):
                    continue  # Skip if not a directory
                
                # Iterate over utterance text files within each dialogue directory
                for utterance_file in os.listdir(dialogue_dir):
                    if utterance_file.endswith(".txt"):
                        utterance_id = utterance_file.split('.')[0]  # Extract utterance identifier
                        
                        # Construct paths for corresponding vision and audio data
                        vision_path = os.path.join(self.base_dir, "MELD_Vision_VIT", emotion, f"{dialogue}_{utterance_id}")
                        vision_files = sorted(glob.glob(os.path.join(vision_path, "*.pt")))
                        
                        audio_path = os.path.join(self.base_dir, "MELD_Speech", emotion, f"{dialogue}_{utterance_id}", "mfcc_features.pt")
                        
                        # Check if vision and audio data exist for the utterance
                        if not os.path.exists(audio_path) or not vision_files:
                            continue  # Skip if corresponding vision or audio data is missing
                        
                        # Append sample information
                        samples.append({
                            'text_path': os.path.join(dialogue_dir, utterance_file),
                            'vision_paths': vision_files,  # List of paths to vision frames
                            'audio_path': audio_path,
                            'label': self.label_mapping[emotion],
                            'emotion': emotion,
                            'dialogue': dialogue,
                            'utterance_id': utterance_id
                        })
        return samples


    def __len__(self):
        return len(self.sample_info)

    def __getitem__(self, idx):
        sample_info = self.sample_info[idx]
        emotion, dialogue, utterance = sample_info['emotion'], sample_info['dialogue'], sample_info['utterance']

        # Load text data
        text_path = os.path.join(self.base_dir, "MELD_Text", emotion, dialogue, f"{utterance}.txt")
        with open(text_path, 'r', encoding='utf-8') as file:
            text_data = file.read()

        # Load vision data (assuming averaging or selecting a specific frame is handled elsewhere)
        vision_path = os.path.join(self.base_dir, "MELD_Vision_VIT", emotion, f"{dialogue}_{utterance}")
        vision_frames = sorted(glob.glob(os.path.join(vision_path, "*.pt")))
        vision_data = [torch.load(frame) for frame in vision_frames]

        # Load audio data
        audio_path = os.path.join(self.base_dir, "MELD_Speech", emotion, dialogue, f"{utterance}.pt")
        audio_data = torch.load(audio_path)

        return {
            'text': text_data,
            'vision': vision_data,  # This is a list of tensors; handling of multiple frames is model-dependent
            'audio': audio_data,
            'label': sample_info['label']
        }
    def save_processed_data(self, idx, processed_data):
        """Saves processed data (e.g., averaged vision frames) to output_dir."""
        if not self.output_dir:
            raise ValueError("Output directory not specified.")
        
        sample_info = self.sample_info[idx]
        emotion, dialogue, utterance = sample_info['emotion'], sample_info['dialogue'], sample_info['utterance']
        
        # Define output path for the processed data
        output_path = os.path.join(self.output_dir, emotion, dialogue, f"{utterance}.pt")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the data
        torch.save(processed_data, output_path)
# Example unified label mapping
unified_label_mapping = {
    "anger": 0,
    "disgust": 1,  # Assuming "disgust" is added here if we're following alphabetical order
    "fear": 2,
    "joy": 3,
    "love": 4,
    "neutral": 5,
    "sadness": 6,
    "surprise": 7,
    "worry": 8  # "worry" might move to 8 if "disgust" is inserted before it
}
file_path = r"F:\FP_multimodal\MELD\MELD-RAW\MELD.Raw\train"
output_dir = r"F:\FP_multimodal\MELD\MELD-RAW\MELD.Raw\train\MELD_train"
MultimodalMELDDataset(base_dir=file_path,label_mapping=unified_label_mapping,output_dir=output_dir)