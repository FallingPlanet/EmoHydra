
import torch
from torch.utils.data import DataLoader, Dataset
from FallingPlanet.orbit.models.multimodal.Hydra import HydraTiny
from FallingPlanet.orbit.models.multimodal.Chimera import Chimera
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from accelerate import Accelerator
from FallingPlanet.orbit.utils.model_utils import MetricsWrapper,EpochMetricsWrapper
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
import os
from FallingPlanet.orbit.models import BertFineTuneTiny
from FallingPlanet.orbit.models import DeitFineTuneTiny
from FallingPlanet.orbit.models.AuTransformer import FPATF_Tiny


if torch.cuda.is_available():
    print("CUDA is available. Proceeding with GPU support.")
    device = torch.device("cuda")
else:
    raise SystemExit("CUDA is not available. Please check your setup.")


class ChimeraInferencePipeline:
    def __init__(self, text_model, vision_model, audio_model, text_state_dict, vision_state_dict, audio_state_dict, text_label_map, vision_label_map, audio_label_map, unified_label_map):
        # Initialize models and load state dictionaries
        
        self.text_model = text_model
        self.vision_model = vision_model
        self.audio_model = audio_model
        
        # Load state dictionaries
        self.text_model.load_state_dict(torch.load(text_state_dict))
        self.vision_model.load_state_dict(torch.load(vision_state_dict))
        self.audio_model.load_state_dict(torch.load(audio_state_dict))

        # Initialize label mappings
        self.text_label_map = text_label_map
        self.vision_label_map = vision_label_map
        self.audio_label_map = audio_label_map
        self.unified_label_map = unified_label_map

    def infer(self, batch):
        text_inputs = {'input_ids': batch['text'][0].to(device), 'attention_mask': batch['text'][1].to(device)}
        vision_inputs = [v.to(device) for v in batch['vision']]
        audio_inputs = batch['audio'].to(device)
        
        # Ensure text inputs are tensors and squeeze if necessary
        text_inputs['input_ids'] = text_inputs['input_ids'].squeeze(0)
        text_inputs['attention_mask'] = text_inputs['attention_mask'].squeeze(0)

        # Convert list of vision input tensors to a batched tensor
        vision_inputs = torch.stack(vision_inputs)  # Stack the inputs
        vision_inputs = vision_inputs.squeeze(1).squeeze(1)
        vision_inputs.to(device)
        # Given shape: [Batch, Channels, Height, Width], aiming to keep this shape
        # Correcting the shape explicitly is done before this step if necessary

        # Model inference
        text_logits = self.text_model(**text_inputs)
        vision_logits_batch = self.vision_model(vision_inputs)
        audio_logits = self.audio_model(audio_inputs)
        
        # Average vision logits across the batch
        # This collapses the batch dimension to yield a single set of logits
        vision_logits_avg = vision_logits_batch.mean(dim=0, keepdim=True)

        # Map logits to labels
        text_labels = self.map_logits_to_labels(text_logits, self.text_label_map)
        
        # Now map the averaged vision logits to labels, resulting in a single vision label
        vision_labels = self.map_logits_to_labels(vision_logits_avg, self.vision_label_map)
        
        audio_labels = self.map_logits_to_labels(audio_logits, self.audio_label_map)
        
        # Fuse outputs (if needed, based on individual logits)
        fused_output = self.fuse_outputs(text_logits, vision_logits_avg, audio_logits)

        return text_labels, vision_labels, audio_labels, fused_output




    def map_logits_to_labels(self, logits, label_map):
        _, indices = torch.max(logits, dim=1)
        unified_dim = max(label_map.values()) + 1  # Assuming continuous values starting from 0
        self.unknown_label = unified_dim - 1  # Assign the last index as unknown label

        labels = []
        for i in indices:
            index = i.item()
            # Check if index is in the label_map, otherwise assign unknown_label
            if index in label_map.values():
                label = list(label_map.keys())[list(label_map.values()).index(index)]
                labels.append(label)
            else:
                labels.append('unknown')  # Or any placeholder for unknown labels
        return labels


    def fuse_outputs(self, text_logits, vision_logits, audio_logits):
        # Example assumes text_logits has 9 classes, and vision_logits and audio_logits have 8 classes
        max_classes = max(text_logits.size(1), vision_logits.size(1), audio_logits.size(1))

        # Function to expand logits tensor to have `max_classes` classes
        def expand_logits(logits, target_size):
           
            if logits.size(1) < target_size:
                diff = target_size - logits.size(1)
                # Assuming logits are on CPU for simplicity; adapt as necessary for GPU tensors
                padding = torch.zeros(logits.size(0), diff)
                logits = torch.cat([logits, padding], dim=1)
               
            return logits

        # Expand each logits tensor to the same size
        text_logits = expand_logits(text_logits, max_classes)
        vision_logits = expand_logits(vision_logits, max_classes)
        audio_logits = expand_logits(audio_logits, max_classes)

        # Now that all logits tensors have the same size, we can safely average them
        fused_output = (text_logits + vision_logits + audio_logits) / 3
        return fused_output

class GPUMetricsWrapper:
    def __init__(self):
        self.correct_predictions = 0
        self.total_predictions = 0

    def update(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        self.correct_predictions += (preds == labels).sum().item()
        self.total_predictions += labels.size(0)

    def compute(self):
        accuracy = self.correct_predictions / self.total_predictions
        return {"accuracy": accuracy}

    def reset(self):
        self.correct_predictions = 0
        self.total_predictions = 0

speech_dict = r"E:\model_saves\EmoSpeak_Transformer_Tinier.pt"
vision_dict = r"D:\Users\WillR\Documents\GitHub\EmoVision\EmoVision_augmented-tiny.pth"
text_dict = r"D:\Users\WillR\Documents\GitHub\EmoBERTv2\EmoBERTv2-tiny.pth"

unified_label_mapping = {
    "anger": 0,
    "angry": 0,
    "disgust": 1,  # Assuming "disgust" is added here if we're following alphabetical order
    "fear": 2,
    "joy": 3,
    "happy": 3,
    "love": 4,
    "neutral": 5,
    "sadness": 6,
    "sad": 6,
    "surprise": 7,
    "worry": 8  # "worry" might move to 8 if "disgust" is inserted before it
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
speech_to_unified = {
    "Neutral": unified_label_mapping["neutral"],
    "Calm": unified_label_mapping["neutral"],
    "Boredom": unified_label_mapping["neutral"],
    "Happy": unified_label_mapping["joy"],
    "Sad": unified_label_mapping["sadness"],
    "Angry": unified_label_mapping["anger"],
    "Fearful": unified_label_mapping["fear"],
    "Disgust": unified_label_mapping["disgust"],
    "Surprise": unified_label_mapping["surprise"],
    # "Love" and "Worry" do not have direct counterparts in this example but could be handled in the model or preprocessing
}
class MultimodalMELDDataset(Dataset):
    def __init__(self, data_dir):
        # data_dir is the directory where your .pt files are saved
        self.data_dir = data_dir
        self.filepaths = self._get_filepaths()

    def _get_filepaths(self):
        # Collect all .pt file paths in the directory
        filepaths = [os.path.join(self.data_dir, fname) for fname in os.listdir(self.data_dir) if fname.endswith('.pt')]
        return filepaths

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        sample = torch.load(filepath)
        return sample
    
# Vision model array to unified label mapping, including "love" and "worry"
# Vision model labels
vision_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise", "love", "worry"]

# Map vision model labels to unified label indices
vision_to_unified = {label: unified_label_mapping[label] for label in vision_labels}






dataset_path = r"F:\FP_multimodal\MELD\MELD_RAW\train\MELD_train"
dataset = MultimodalMELDDataset(data_dir=dataset_path)

def custom_collate_fn(batch):
        # Extract sequences and their lengths
        text_data = [item['text'] for item in batch]
        vision_data = [item['vision'] for item in batch]
        audio_data = [item['audio'] for item in batch]
        labels = [item['label'] for item in batch]

        # Pad text sequences
        text_input_ids = pad_sequence([x[0] for x in text_data], batch_first=True)
        text_attention_masks = pad_sequence([x[1] for x in text_data], batch_first=True)

        # Handle vision and audio data normally, assuming they're already of consistent size or don't require padding
        vision_batch = default_collate(vision_data)
        audio_batch = default_collate(audio_data)

        return {
            'text': (text_input_ids, text_attention_masks),
            'vision': vision_batch,
            'audio': audio_batch,
            'label': torch.tensor(labels)
        }
inference_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

# Assuming you have already initialized the pre-trained models
text_model = BertFineTuneTiny(num_labels=[9]).to(device)
vision_model = DeitFineTuneTiny(num_labels=[8]).to(device)
audio_model = FPATF_Tiny(target_channels=30, num_classes=7, num_heads=16, dim_feedforward=1024, num_layers=4, dropout=0.1).to(device)

# Load the pre-trained state dictionaries
text_state_dict = r"D:\Users\WillR\Documents\GitHub\EmoBERTv2\EmoBERTv2-tiny.pth"
vision_state_dict = r"D:\Users\WillR\Documents\GitHub\EmoVision\EmoVision_augmented-tiny.pth"
audio_state_dict = r"E:\model_saves\EmoSpeak_Transformer_Tinier.pt"

# Initialize the inference pipeline
pipeline = ChimeraInferencePipeline(
    text_model, vision_model, audio_model,
    text_state_dict=text_state_dict,
    vision_state_dict=vision_state_dict,
    audio_state_dict=audio_state_dict,
    text_label_map=text_label_mapping,
    vision_label_map=vision_to_unified,
    audio_label_map=speech_to_unified,
    unified_label_map=unified_label_mapping
)


correct_predictions = 0
top2_predictions = 0
total_predictions = 0

# Initialize the metrics wrapper
metrics_wrapper = GPUMetricsWrapper()

# Ensure your DataLoader, models, and tensors are all set to use the GPU as shown earlier

# Evaluation loop
for batch in inference_loader:
    # Ensure batch data is moved to GPU
    inputs = {'text': (batch['text'][0].to(device), batch['text'][1].to(device)), 
              'vision': [v.to(device) for v in batch['vision']],
              'audio': batch['audio'].to(device)}
    labels = batch['label'].to(device)
    
    # Perform inference
    _, _, _, fused_output = pipeline.infer(inputs)
    
    # Update metrics
    metrics_wrapper.update(fused_output, labels)

# Compute and display metrics
metrics_results = metrics_wrapper.compute()
print("Evaluation Results:", metrics_results)

# Reset for next evaluation, if necessary
metrics_wrapper.reset()

