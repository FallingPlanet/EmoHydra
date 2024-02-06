import cv2
import os
import pandas as pd
from transformers import ViTFeatureExtractor
from PIL import Image

# Initialize the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

csv_path = r"F:\FP_multimodal\MELD\MELD-RAW\MELD.Raw\train\train_sent_emo.csv"
video_directory = r"F:\FP_multimodal\MELD\MELD-RAW\MELD.Raw\train\train_splits"
output_directory = r"F:\FP_multimodal\MELD\MELD-RAW\MELD.Raw\train\MELD_Vision_VIT"
df = pd.read_csv(csv_path)
# Create the output directory if it does not exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def preprocess_video_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps / 4))  # Ensure at least 1 to avoid division by zero
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(1)) % frame_interval == 0:
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = feature_extractor(images=frame_pil, return_tensors="pt")
            
            # Assuming you want to save the preprocessed PIL images
            frame_save_path = os.path.join(output_folder, f"frame_{frame_count}.png")
            frame_pil.save(frame_save_path)
            frame_count += 1

    cap.release()

# Process videos and save preprocessed frames
running_index = 0  # Start from 0 assuming DataFrame is 0-indexed
for filename in sorted(os.listdir(video_directory)):
    if filename.endswith(".mp4"):
        video_path = os.path.join(video_directory, filename)
        video_output_folder = os.path.join(output_directory, f"video_{running_index}")
        
        # Create a specific output folder for each video
        if not os.path.exists(video_output_folder):
            os.makedirs(video_output_folder)
        
        emotion_label = df.iloc[running_index]['Emotion']
        preprocess_video_frames(video_path, video_output_folder)
        # Preprocessed frames for each video are saved in its specific output folder
        running_index += 1  # Move to the next index for the next iteration



