import pickle
data_path = r"F:\FP_multimodal\IEMOCAP_full_release\IEMOCAP_features.pkl"

# users should use this instructor to load pkl dataset. 
videoIDs, videoSpeakers, videoLabels, videoText,\
    videoAudio, videoVisual, videoSentence, trainVid,\
        testVid = pickle.load(open(data_path, 'rb'), encoding='latin1')
import pickle
data_path = r"F:\FP_multimodal\IEMOCAP_full_release\IEMOCAP_features.pkl"

# users should use this instructor to load pkl dataset. 
videoIDs, videoSpeakers, videoLabels, videoText,\
    videoAudio, videoVisual, videoSentence, trainVid,\
        testVid = pickle.load(open(data_path, 'rb'), encoding='latin1')
print(videoSpeakers['Ses03M_impro08b'], '\n')
print(videoLabels['Ses03M_impro08b'], '\n')
print(len(videoText['Ses03M_impro08b']), videoText['Ses03M_impro08b'][0].shape,'\n')
print(len(videoAudio['Ses03M_impro08b']), videoAudio['Ses03M_impro08b'][0].shape, '\n')
print(len(videoVisual['Ses03M_impro08b']), videoVisual['Ses03M_impro08b'][0].shape, '\n')
print(videoSentence['Ses03M_impro08b'], '\n')

import torch
from transformers import BertTokenizer, ViTFeatureExtractor
import torchaudio.transforms as T
import numpy as np
import pickle

# Define the MFCC transformation
mfcc_transform = T.MFCC(sample_rate=16000, n_mfcc=30, melkwargs={'n_fft': 199, 'n_mels': 30, 'hop_length': 160, 'win_length': 199})

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Initialize ViT feature extractor
vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

def process_audio(audio_data):
    # Convert to PyTorch tensor and apply MFCC
    mfcc_features = [mfcc_transform(torch.tensor(audio, dtype=torch.float32).unsqueeze(0)) for audio in audio_data]
    return mfcc_features

def process_visual(visual_data):
    # Apply ViT feature extraction
    visual_features = [vit_feature_extractor(images, return_tensors="pt") for images in visual_data]
    return visual_features

def process_entry(videoID, videoAudio, videoVisual, videoSentence, tokenizer, vit_feature_extractor):
    # Process text data
    text_features = [tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512) for sentence in videoSentence[videoID]]
    
    # Process audio data
    audio_features = process_audio(videoAudio[videoID])
    
    # Process visual data
    visual_features = process_visual(videoVisual[videoID])
    
    return {'text': text_features, 'audio': audio_features, 'visual': visual_features}

# Load your data
with open(data_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = data

# Process each entry
processed_data = {videoID: process_entry(videoID, videoAudio, videoVisual, videoSentence, tokenizer, vit_feature_extractor) for videoID in videoIDs}

# Save processed data
torch.save(processed_data, "F:\FP_multimodal\IEMOCAP_full_release\processed_IEMOCAP.pt")


