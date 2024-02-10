import torchaudio
import moviepy.editor as mp
import pandas as pd
import os
import torch

csv_path = r"F:\FP_multimodal\MELD\MELD-RAW\MELD.Raw\train\train_sent_emo.csv"
video_directory = r"F:\FP_multimodal\MELD\MELD-RAW\MELD.Raw\train\train_splits"
output_directory = r"F:\FP_multimodal\MELD\MELD-RAW\MELD.Raw\train\MELD_Speech"
df = pd.read_csv(csv_path)

# Create the output directory if it does not exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
def extract_audio_from_video(video_path):
    """
    Extracts the audio track from the video file and loads it as a waveform.
    """
    # Extract audio from video
    audio_clip = mp.AudioFileClip(video_path)
    audio_clip.write_audiofile("temp_audio.wav", codec='pcm_s16le')
    audio_clip.close()
    
    # Load the waveform from the extracted audio
    waveform, sample_rate = torchaudio.load("temp_audio.wav")
    
    # Clean up the temporary audio file
    os.remove("temp_audio.wav")
    
    return waveform, sample_rate

def extract_mfccs(waveform, sample_rate, n_mfcc=30, win_length=400, hop_length=160, n_mels=30, n_fft=1024):
    """
    Extracts MFCC features from the waveform of an audio signal.
    """
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            'win_length': win_length, 
            'hop_length': hop_length, 
            'n_mels': n_mels,
            'n_fft': n_fft
        }
    )
    mfcc = mfcc_transform(waveform)
    return mfcc

for idx, row in df.iterrows():
    video_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
    video_path = os.path.join(video_directory, video_filename)
    emotion_label = row['Emotion']
    mfcc_output_folder = os.path.join(output_directory, emotion_label, video_filename[:-4])  # Removing .mp4 extension

    if not os.path.exists(mfcc_output_folder):
        os.makedirs(mfcc_output_folder)
    
    if os.path.exists(video_path):
        waveform, sample_rate = extract_audio_from_video(video_path)
        mfcc_features = extract_mfccs(waveform, sample_rate)
        
        # Save the MFCC features as a tensor
        mfcc_save_path = os.path.join(mfcc_output_folder, "mfcc_features.pt")
        torch.save(mfcc_features, mfcc_save_path)
