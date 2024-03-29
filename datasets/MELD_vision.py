import cv2
import os
import pandas as pd
import torch
from PIL import Image
from transformers import ViTFeatureExtractor
from yolov5 import YOLOv5

# Initialize the YOLOv5 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLOv5(r"D:\Users\WillR\Documents\GitHub\EmoHydra\yolov5s.pt", device="cpu")  # Load the pre-trained "small" model

# Initialize the ViT feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

def preprocess_and_save(video_path, output_folder, frame_count=0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps / 2))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame_number % frame_interval == 0:
            # Convert BGR frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Perform inference
            results = model.predict(rgb_frame)
            detections = results.pandas().xyxy[0]  # Extract predictions

            if len(detections) > 0:  # Check if any faces are detected
                for index, det in detections.iterrows():
                    # Assuming class 0 is 'person', adjust as per your model's classes
                    if det['class'] == 0:
                        # Crop detected face based on bounding box coordinates
                        x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                        face = Image.fromarray(rgb_frame[y1:y2, x1:x2])
                        
                        # Extract features using ViT
                        inputs = feature_extractor(images=face, return_tensors="pt")["pixel_values"].to(device)
                        
                        # Save the tensor
                        tensor_save_path = os.path.join(output_folder, f"frame_{frame_count}.pt")
                        torch.save(inputs.cpu(), tensor_save_path)
                        frame_count += 1

    cap.release()

# Setup directories
csv_path = 'F:\\FP_multimodal\\MELD\\MELD.Raw\\train\\train_sent_emo.csv'
video_directory = 'F:\\FP_multimodal\\MELD\\MELD.Raw\\train\\train_splits'
output_directory = 'F:\\FP_multimodal\\MELD\\MELD.Raw\\train\\MELD_Vision_VIT'

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

df = pd.read_csv(csv_path)

# Process videos and save preprocessed frames sequentially
for idx, row in df.iterrows():
    video_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
    video_path = os.path.join(video_directory, video_filename)
    emotion_label = row['Emotion']
    video_output_folder = os.path.join(output_directory, emotion_label, video_filename[:-4])

    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)
    
    if os.path.exists(video_path):
        preprocess_and_save(video_path, video_output_folder)
    else:
        print(f"Video file at {video_path} not found.")




