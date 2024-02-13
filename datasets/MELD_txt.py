import pandas as pd
import os

# Paths
csv_path = r"F:\FP_multimodal\MELD\MELD.Raw\train\train_sent_emo.csv"
output_directory = r"F:\FP_multimodal\MELD\MELD.Raw\train\MELD_Text"

# Load the CSV
df = pd.read_csv(csv_path)

# Create the output directory if it does not exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Function to save text utterances
def save_text_utterances(row, output_folder):
    utterance_text = row['Utterance']
    emotion_label = row['Emotion']
    dialogue_id = row['Dialogue_ID']
    utterance_id = row['Utterance_ID']
    
    # Constructing folder path for the emotion
    emotion_folder = os.path.join(output_folder, emotion_label)
    if not os.path.exists(emotion_folder):
        os.makedirs(emotion_folder)
    
    # Adjusted to save directly under the emotion folder, naming files to include both dialogue and utterance IDs
    text_file_path = os.path.join(emotion_folder, f"dia{dialogue_id}_utt{utterance_id}.txt")
    
    # Save the utterance text to a file
    with open(text_file_path, 'w', encoding='utf-8') as f:
        f.write(utterance_text)

# Iterate over the DataFrame and save each utterance
for idx, row in df.iterrows():
    save_text_utterances(row, output_directory)

