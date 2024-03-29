import torch
from torch.utils.data import Dataset
import os
import glob
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MultimodalMELDDataset(Dataset):
    def __init__(self, base_dir, label_mapping, output_dir=None):
        self.base_dir = base_dir
        self.label_mapping = label_mapping
        self.output_dir = output_dir
        self.sample_info = self._gather_samples_info()

    def _gather_samples_info(self):
      
        missing_data_count = {'text': 0, 'vision': 0, 'audio': 0, 'total': 0}
        samples = []
        # Base directories for text, vision, and audio
        text_base_dir = os.path.join(self.base_dir, "MELD_Text")
        vision_base_dir = os.path.join(self.base_dir, "MELD_Vision_VIT")
        audio_base_dir = os.path.join(self.base_dir, "MELD_Speech")

        for emotion in os.listdir(text_base_dir):
            emotion_dir = os.path.join(text_base_dir, emotion)
            if not os.path.isdir(emotion_dir):
                continue

            # Adjusted to directly iterate through .txt files representing utterances
            for text_file in glob.glob(os.path.join(emotion_dir, "*.txt")):
                file_parts = os.path.splitext(os.path.basename(text_file))[0].split('_')
                dialogue_id = file_parts[0]  # e.g., "dia12"
                utterance_id = file_parts[1]  # e.g., "utt2"
                
                # Construct paths for vision and audio files
                vision_path = os.path.join(vision_base_dir, emotion, f"{dialogue_id}_{utterance_id}")
                vision_files = sorted(glob.glob(os.path.join(vision_path, "*.pt")))
                
                audio_path = os.path.join(audio_base_dir, emotion, f"{dialogue_id}_{utterance_id}", "mfcc_features.pt")
                
                if os.path.exists(audio_path) and vision_files:
                    samples.append({
                        'text_path': text_file,
                        'vision_paths': vision_files,
                        'audio_path': audio_path,
                        'label': self.label_mapping[emotion],
                        'emotion': emotion,
                        'dialogue': dialogue_id,
                        'utterance_id': utterance_id
                    })
                else:
                
                    missing_data_count['total'] += 1
                    if not os.path.exists(audio_path): missing_data_count['audio'] += 1
                    if not vision_files: missing_data_count['vision'] += 1
                    logging.warning(f"Missing data for {dialogue_id}_{utterance_id} in emotion {emotion}. Missing types: {'audio' if not os.path.exists(audio_path) else ''} {'vision' if not vision_files else ''}")

        logging.info(f"Total missing samples: {missing_data_count['total']}. Details - Audio: {missing_data_count['audio']}, Vision: {missing_data_count['vision']}")

        return samples

       
    def __len__(self):
        return len(self.sample_info)

    def __getitem__(self, idx):
        sample_info = self.sample_info[idx]
        with open(sample_info['text_path'], 'r', encoding='utf-8') as file:
            text_data = file.read()
        vision_data = [torch.load(frame) for frame in sample_info['vision_paths']]
        audio_data = torch.load(sample_info['audio_path'])
        
        return {
            'text': text_data,
            'vision': vision_data,
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
file_path = r"F:\FP_multimodal\MELD\MELD.Raw\train"
output_dir = r"F:\FP_multimodal\MELD\MELD.RAW\train\MELD_train"
MultimodalMELDDataset(base_dir=file_path,label_mapping=unified_label_mapping,output_dir=output_dir)

# Instantiate the dataset
dataset = MultimodalMELDDataset(base_dir=file_path, label_mapping=unified_label_mapping, output_dir=output_dir)

for idx in range(len(dataset)):
    sample = dataset[idx]  # Load the data
    
    # Define the output path for the entire sample
    output_path = os.path.join(dataset.output_dir, f"sample_{idx}.pt")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the entire sample as a .pt file
    torch.save(sample, output_path)
    
