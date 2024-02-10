import cv2
import os
import pandas as pd
from transformers import ViTFeatureExtractor
from PIL import Image
import torch

# Initialize the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

csv_path = r"F:\FP_multimodal\MELD\MELD-RAW\MELD.Raw\train\train_sent_emo.csv"
video_directory = r"F:\FP_multimodal\MELD\MELD-RAW\MELD.Raw\train\train_splits"
output_directory = r"F:\FP_multimodal\MELD\MELD-RAW\MELD.Raw\train\MELD_Vision_VIT"
df = pd.read_csv(csv_path)

# Create the output directory if it does not exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def preprocess_and_save(video_path, output_folder, frame_count):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps / 2))  # Ensure at least 1 to avoid division by zero

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(1)) % frame_interval == 0:
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = feature_extractor(images=frame_pil, return_tensors="pt")["pixel_values"]
            
            # Save the tensor and label
            tensor_save_path = os.path.join(output_folder, f"frame_{frame_count}.pt")
            torch.save(inputs, tensor_save_path)
            frame_count += 1

    cap.release()

# Process videos and save preprocessed frames sequentially
for idx, row in df.iterrows():
    video_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
    video_path = os.path.join(video_directory, video_filename)
    emotion_label = row['Emotion']
    video_output_folder = os.path.join(output_directory, emotion_label, video_filename[:-4])  # Remove .mp4

    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)
    
    if os.path.exists(video_path):
        preprocess_and_save(video_path, video_output_folder, 0)  # Start frame counting from 0 for each video
    else:
        print(f"Video file at {video_path} not found.")



