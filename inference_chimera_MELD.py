
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
from torchmetrics import Accuracy, Precision, Recall, F1Score, MatthewsCorrCoef

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
        
        # Model inference
        text_logits = self.text_model(**text_inputs)
        vision_logits_batch = self.vision_model(vision_inputs.to(device))  # Move vision_inputs to GPU
        audio_logits = self.audio_model(audio_inputs)
        
        # Debugging: Initial logits
        print("Initial Logits:")
        print(f"Text logits: {text_logits}")
        print(f"Vision logits batch: {vision_logits_batch}")
        print(f"Audio logits: {audio_logits}")

        # Average vision logits across the batch if necessary
        vision_logits_avg = vision_logits_batch.mean(dim=0, keepdim=True)

        # Remap logits to align with the unified mapping
        text_logits_remapped = self.remap_logits_to_unified(text_logits, self.text_label_map, self.unified_label_map)
        vision_logits_remapped = self.remap_logits_to_unified(vision_logits_avg, self.vision_label_map, self.unified_label_map)
        audio_logits_remapped = self.remap_logits_to_unified(audio_logits, self.audio_label_map, self.unified_label_map)

        # Debugging: Logits before and after remapping
        print("Remapped Logits:")
        print(f"Text logits remapped: {text_logits_remapped}")
        print(f"Vision logits remapped: {vision_logits_remapped}")
        print(f"Audio logits remapped: {audio_logits_remapped}")

        # Convert remapped logits to labels (or proceed with logits for fusion or other processing)
        text_labels, text_max_indices = self.map_logits_to_labels(text_logits_remapped, self.unified_label_map, return_indices=True)
        vision_labels, vision_max_indices = self.map_logits_to_labels(vision_logits_remapped, self.unified_label_map, return_indices=True)
        audio_labels, audio_max_indices = self.map_logits_to_labels(audio_logits_remapped, self.unified_label_map, return_indices=True)

        # Debugging: Labels and indices from each model
        print("Labels and max indices from each model:")
        print(f"Text labels and indices: {list(zip(text_labels, text_max_indices))}")
        print(f"Vision labels and indices: {list(zip(vision_labels, vision_max_indices))}")
        print(f"Audio labels and indices: {list(zip(audio_labels, audio_max_indices))}")

        # Fuse outputs if needed, based on remapped logits
        fused_output = self.fuse_outputs(text_logits_remapped, vision_logits_remapped, audio_logits_remapped)

        _, predicted_indices = torch.max(fused_output, dim=1)
        print(f"Fused output: {fused_output}")
        for i, index in enumerate(predicted_indices):
            predicted_label = self.index_to_label(index.item())
            

        return text_labels, vision_labels, audio_labels, fused_output






    def remap_logits_to_unified(self, logits, current_mapping, unified_mapping):
        # Determine the maximum index in the unified mapping to know the required size of the new logits tensor
        max_index_unified = max(unified_mapping.values())
        
        # Expand the logits tensor to match the size required by the unified mapping
        # Note: logits.shape[-1] gives the current class dimension size, and we assume logits is 2D [batch_size, num_classes]
        required_size = max_index_unified + 1  # +1 because index starts at 0
        if logits.shape[-1] < required_size:
            # Calculate the difference and pad accordingly
            padding_size = required_size - logits.shape[-1]
            padding = torch.zeros(logits.shape[0], padding_size, device=logits.device)
            logits = torch.cat([logits, padding], dim=-1)
        
        # Initialize an empty tensor for the new logits arrangement with the expanded shape
        new_logits = torch.zeros_like(logits)
        
        # Remap the logits based on the current and unified mappings
        for label, index in current_mapping.items():
            # Normalize the label to ensure consistency with the unified label mapping
            
            if label in unified_mapping:
                unified_index = unified_mapping[label]
                new_logits[..., unified_index] = logits[..., index]
            
        return new_logits


    

    def index_to_label(self, index):
        # Assuming you have a reverse mapping from index to label name
        index_to_label_map = {value: key for key, value in self.unified_label_map.items()}
        return index_to_label_map.get(index, "unknown")

    def map_logits_to_labels(self, logits, label_map, return_indices=False):
        _, indices = torch.max(logits, dim=1)
        labels = []
        for index in indices:
            label_name = self.index_to_label(index.item())  # Convert index to corresponding label name
            label = label_map.get(label_name, 'unknown')
            labels.append(label)
        if return_indices:
            return labels, indices.tolist()
        return labels




    def fuse_outputs(self, text_logits, vision_logits, audio_logits):
        # Example assumes text_logits has 9 classes, and vision_logits and audio_logits have 8 classes
        max_classes = max(text_logits.size(1), vision_logits.size(1), audio_logits.size(1))

        # Function to expand logits tensor to have `max_classes` classes
        def expand_logits(logits, target_size):
            if logits.size(1) < target_size:
                diff = target_size - logits.size(1)
                padding = torch.zeros(logits.size(0), diff).to(device)
                logits = torch.cat([logits, padding], dim=1).to(device)
            return logits

        # Expand each logits tensor to the same size
        text_logits_expanded = expand_logits(text_logits, max_classes)
        vision_logits_expanded = expand_logits(vision_logits.to(device), max_classes)
        audio_logits_expanded = expand_logits(audio_logits, max_classes)

        

        # Now that all logits tensors have the same size, we can safely average them
        fused_output = (text_logits_expanded + vision_logits_expanded + audio_logits_expanded) / 3

        

        return fused_output



class GPUMetricsWrapper:
    def __init__(self, num_classes, device):
        self.device = device
        # Existing metrics
        self.accuracy = Accuracy(top_k=1, task='multiclass', num_classes=num_classes).to(device)
        self.top2_accuracy = Accuracy(top_k=2, task='multiclass', num_classes=num_classes).to(device)
        self.precision = Precision(num_classes=num_classes, average='macro', task='multiclass').to(device)
        self.recall = Recall(num_classes=num_classes, average='macro', task='multiclass').to(device)
        self.f1_score = F1Score(num_classes=num_classes, average='macro', task='multiclass').to(device)
        # New metrics
        self.mcc = MatthewsCorrCoef(num_classes=num_classes,task ='multiclass').to(device)
        self.weighted_f1_score = F1Score(num_classes=num_classes, average='weighted', task='multiclass').to(device)

    def update(self, outputs, labels):
        # Update metrics
        self.accuracy(outputs, labels)
        self.top2_accuracy(outputs, labels)
        self.precision(outputs, labels)
        self.recall(outputs, labels)
        self.f1_score(outputs, labels)
        self.mcc(outputs, labels)
        self.weighted_f1_score(outputs, labels)

    def compute(self):
        # Compute and return metrics
        return {
            "accuracy": self.accuracy.compute().item(),
            "top2_accuracy": self.top2_accuracy.compute().item(),
            "precision": self.precision.compute().item(),
            "recall": self.recall.compute().item(),
            "f1_score": self.f1_score.compute().item(),
            "mcc": self.mcc.compute().item(),
            "weighted_f1_score": self.weighted_f1_score.compute().item()
        }

    def reset(self):
        # Reset metrics
        self.accuracy.reset()
        self.top2_accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1_score.reset()
        self.mcc.reset()
        self.weighted_f1_score.reset()


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
speech_label_mapping = {
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
    vision_label_map=vision_label_mapping,
    audio_label_map=speech_label_mapping,
    unified_label_map=unified_label_mapping
)



# Initialize the metrics wrapper
metrics_wrapper = GPUMetricsWrapper(num_classes=9,device=device)

# Ensure your DataLoader, models, and tensors are all set to use the GPU as shown earlier

# Evaluation loop
for batch_idx, batch in enumerate(inference_loader):
    # Ensure batch data is moved to GPU
    inputs = {
        'text': (batch['text'][0].to(device), batch['text'][1].to(device)),
        'vision': [v.to(device) for v in batch['vision']],
        'audio': batch['audio'].to(device)
    }
    labels = batch['label'].to(device)

    # Perform inference
    text_labels, vision_labels, audio_labels, fused_output = pipeline.infer(inputs)

    # Get the predictions from fused_output
    _, predicted_indices = torch.max(fused_output, dim=1)
    
    # Update metrics
    metrics_wrapper.update(fused_output, labels)

    # Print the original and predicted labels for inspection
    print(f"Batch {batch_idx + 1}:")
    for i, (original_label_index, predicted_index) in enumerate(zip(labels, predicted_indices)):
        original_label = pipeline.index_to_label(original_label_index.item())
        predicted_label = pipeline.index_to_label(predicted_index.item())
        print(f"Sample {i}: Original Label: {original_label}, Predicted Label: {predicted_label}")

# Compute and display metrics
metrics_results = metrics_wrapper.compute()
print("Evaluation Results:", metrics_results)

# Reset for next evaluation, if necessary
metrics_wrapper.reset()


