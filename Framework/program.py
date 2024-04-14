import cv2
import torch
import sounddevice as sd
from transformers import pipeline, BertTokenizer, ViTFeatureExtractor, ViTModel
from pathlib import Path
from FallingPlanet.orbit.models.multimodal.Hydra import HydraTinyRefactored
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

# Initialize the model
model_path = Path("path_to_trained_model.pth")
num_classes = len(unified_label_mapping)
model =HydraTinyRefactored(num_classes=num_classes, requires_grad=False,text_label_map=text_label_mapping,audio_label_map=audio_label_mapping,vision_label_map=vision_label_mapping,unified_label_map=unified_label_mapping)
model.load_state_dict(torch.load(model_path))

# Load BERT tokenizer and ViT feature extractor
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')

# Initialize speech recognition
whisper_processor = pipeline("automatic-speech-recognition", model="openai/whisper-small")

def capture_and_process_audio(duration=1, samplerate=16000):
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    transcribed_text = whisper_processor(audio)["text"]
    inputs = tokenizer(transcribed_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return inputs['input_ids'], inputs['attention_mask']

def capture_and_process_video():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            inputs = feature_extractor(images=face, return_tensors="pt")
            outputs = vit_model(**inputs)
            vision_features = outputs.last_hidden_state
            return vision_features

    cap.release()
    return None

def real_time_processing(model):
    cap = cv2.VideoCapture(0)  # Start video capture

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vision_features = process_frame(frame)  # Process video frame
        input_ids, attention_mask = capture_and_process_audio()  # Process audio and transcribe text
        audio_features = None  # Placeholder for audio feature extraction

        # Check if all modalities are available
        if vision_features is not None and audio_features is not None:
            # Model prediction
            probabilities = model(input_ids, attention_mask, vision_features, audio_features)
            print("Predicted probabilities:", probabilities)

    cap.release()

# Load the model and run the real-time processing
real_time_processing(model)
