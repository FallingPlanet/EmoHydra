import pandas as pd
import os

# Paths
csv_path = r"F:\FP_multimodal\MELD\MELD-RAW\MELD.Raw\train\train_sent_emo.csv"
output_directory = r"F:\FP_multimodal\MELD\MELD-RAW\MELD.Raw\train\MELD_Text"

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
    
    # Constructing folder path for the emotion and dialogue
    emotion_folder = os.path.join(output_folder, emotion_label)
    dialogue_folder = os.path.join(emotion_folder, f"dia{dialogue_id}")
    
    if not os.path.exists(dialogue_folder):
        os.makedirs(dialogue_folder)
    
    # File path for the utterance text
    text_file_path = os.path.join(dialogue_folder, f"utt{utterance_id}.txt")
    
    # Save the utterance text to a file
    with open(text_file_path, 'w', encoding='utf-8') as f:
        f.write(utterance_text)

# Iterate over the DataFrame and save each utterance
for idx, row in df.iterrows():
    save_text_utterances(row, output_directory)
