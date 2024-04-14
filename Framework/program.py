import cv2
import torch
import sounddevice as sd
from transformers import pipeline, BertTokenizerFast, ViTFeatureExtractor, ViTModel
from pathlib import Path
from FallingPlanet.orbit.models.multimodal.Hydra import HydraTiny
speech_dict = r"E:\model_saves\EmoSpeak_Transformer_Tinier.pt"
vision_dict = r"D:\Users\WillR\Documents\GitHub\EmoVision\EmoVision_augmented-tiny.pth"
text_dict = r"D:\Users\WillR\Documents\GitHub\EmoBERTv2\EmoBERTv2-tiny.pth"

unified_label_mapping = {
    "anger": 0, "Anger": 0,
    "angry": 0, "Angry": 0,
    "disgust": 1, "Disgust": 1,
    "fear": 2, "Fear": 2, "Fearful": 2,"fearful": 2,
    "joy": 3, "Joy": 3,
    "happy": 3, "Happy": 3,
    "love": 4, "Love": 4,
    "neutral": 5, "Neutral": 5,"Calm": 5, "calm": 5, "Boredom": 5, "boredom": 5,
    "sadness": 6, "Sadness": 6,
    "sad": 6, "Sad": 6,
    "surprise": 7, "Surprise": 7,
    "worry": 8, "Worry": 8
}

# Adjusted unified label mapping to include all emotions
text_label_mapping = {
    "anger": 0,
    "angry": 0,
    "disgust": 8,
    "fear": 1,
    "joy": 2,
    "happy": 2,
    "love": 3,  # Assuming "love" can be treated uniquely or mapped closely to positive emotions like "joy"
    "neutral": 4,
    "sadness": 5,
    "sad": 5,
    "surprise": 6,
    "worry": 7  # Assigning "worry" a unique value as it represents a distinct emotion
}

# Speech model dictionary to unified label mapping, with a comprehensive approach
vision_label_mapping = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6
}
audio_label_mapping = {
    "Neutral": 0,
    "Calm": 0,
    "Boredom": 0,
    "Happy": 1,
    "Sad": 2,
    "Angry": 3,
    "Fearful": 4,
    "Disgust": 5,
    "Surprise": 6
}

import cv2
import torch
import sounddevice as sd
import numpy as np
from transformers import BertTokenizer, ViTFeatureExtractor
from yolov5 import YOLOv5
from pathlib import Path
from FallingPlanet.orbit.models.multimodal.Hydra import HydraTiny
import librosa
# Load the pre-trained models and components
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model = YOLOv5(r"D:\Users\WillR\Documents\GitHub\EmoHydra\yolov5s.pt")
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
whisper_processor = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# Initialize and load the multimodal model


model = HydraTiny(num_classes=9, requires_grad=False,text_label_map=text_label_mapping,audio_label_map=audio_label_mapping,vision_label_map=vision_label_mapping,unified_label_map=unified_label_mapping,mode="concat").to(device)
model.load_modal_state_dicts(text_dict=text_dict, audio_dict=speech_dict, vision_dict=vision_dict)

def capture_and_process_audio(duration=.30, samplerate=16000):
    # Record the audio
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Wait for the recording to finish
    audio = audio.squeeze()  # Ensure it's mono

    # Generate MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=samplerate, n_mfcc=30, n_fft=2048, hop_length=512, win_length=1024)
    mfccs_tensor = torch.tensor(mfccs).unsqueeze(0).unsqueeze(0).float()  # Add a batch dimension

    # Transcription using Whisper
    transcribed_text = whisper_processor(audio)["text"]
    inputs = tokenizer(transcribed_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids, attention_mask = inputs['input_ids'].to(device), inputs['attention_mask'].to(device)

    return (input_ids, attention_mask), mfccs_tensor.to(device), transcribed_text


def process_video_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolo_model.predict(frame_rgb)
    detections = results.pandas().xyxy[0]  # Extract predictions

    vision_features = []
    for index, det in detections.iterrows():
        if det['class'] == 0:  # Assuming class 0 is 'person', adjust as needed
            x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
            face = frame_rgb[y1:y2, x1:x2]
            face = cv2.resize(face, (224, 224))  # Resize to match ViT input size
            inputs = feature_extractor(images=face, return_tensors="pt")
            vision_features.append(inputs['pixel_values'].squeeze(0).to(device))

    if vision_features:
        vision_features = torch.stack(vision_features)  # Stack features if multiple faces
    return vision_features

import cv2
import numpy as np

def real_time_processing(model):
    cap = cv2.VideoCapture(0)  # Start video capture

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vision_features = process_video_frame(frame)
        (input_ids, attention_mask), audio_features, transcription = capture_and_process_audio()

        if vision_features is not None:
            probabilities = model((input_ids, attention_mask), vision_features, audio_features)
            print("Predicted probabilities:", probabilities)
            print("Transcription:", transcription)  # Now using the returned transcription
            
            # Visualize the highest probability predictions and bounding boxes
            results = yolo_model.predict(frame)
            for det in results.xyxy[0]:
                xmin, ymin, xmax, ymax, conf, cls = int(det[0]), int(det[1]), int(det[2]), int(det[3]), det[4], det[5]
                if int(cls) in vision_label_mapping:
                    label = f'{vision_label_mapping[int(cls)]}: {conf:.2f}'
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                    cv2.putText(frame, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow("Camera Output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
real_time_processing(model)

