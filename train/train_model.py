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

class MultimodalClassifier:
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset, num_classes: int, batch_size: int = 1, learning_rate: float = 0.001, text_state_dict: str = "", vision_state_dict: str = "", audio_state_dict: str = "", mode: str = 'macro', vision_label_map: dict = None, text_label_map: dict = None, audio_label_map: dict = None,unified_label_map: dict = None):
        self.accelerator = Accelerator()

        # Initialize the model with state dictionaries for each modality
        self.model = HydraTiny(num_classes=num_classes, feature_dim=30, requires_grad=True,text_label_map=text_label_map,audio_label_map=audio_label_map,vision_label_map=vision_label_map,unified_label_map=unified_label_mapping)
        self.model.load_modal_state_dicts(text_dict=text_state_dict, audio_dict=audio_state_dict, vision_dict=vision_state_dict)
        self.loss_function = CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=self.custom_collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,collate_fn=self.custom_collate_fn)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,collate_fn=self.custom_collate_fn)

        self.model, self.optimizer, self.train_loader, self.val_loader, self.test_loader = self.accelerator.prepare(
        self.model, self.optimizer, self.train_loader, self.val_loader, self.test_loader)

        self.epoch_metrics = EpochMetricsWrapper(num_classes=num_classes, device=self.accelerator.device, mode=mode)
        

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        self.epoch_metrics.reset()

        for batch_idx, sample in enumerate(tqdm(self.train_loader, desc="Training", position=0, leave=True)):
            input_ids, attention_masks, vision_data, audio_data, labels = self.prepare_data(sample)
            text_data = (input_ids, attention_masks)

            self.optimizer.zero_grad()
            outputs, vision_labels, text_labels, audio_labels = self.model(vision_data, text_data, audio_data)
            loss = self.loss_function(outputs, labels)
            self.accelerator.backward(loss)
            self.optimizer.step()

            running_loss += loss.item()
            self.epoch_metrics.update(outputs.detach(), labels)

        avg_loss = running_loss / len(self.train_loader)
        epoch_metrics = self.epoch_metrics.compute_epoch_metrics()
        if self.accelerator.is_local_main_process:
            tqdm.write(f"Training Loss: {avg_loss:.4f}, Epoch Metrics: {epoch_metrics}")
        self.epoch_metrics.reset()

        return avg_loss, epoch_metrics




    def custom_collate_fn(self,batch):
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
    def run_evaluation(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        self.epoch_metrics.reset()

        for sample in tqdm(data_loader, desc="Evaluating", position=0, leave=True, disable=not self.accelerator.is_local_main_process):
            input_ids, attention_masks, vision_data, audio_data, labels = self.prepare_data(sample)
            text_data = (input_ids, attention_masks)

            outputs, vision_labels, text_labels, audio_labels = self.model(vision_data, text_data, audio_data)
            loss = self.loss_function(outputs, labels)

            total_loss += loss.item()
            self.epoch_metrics.update(outputs.detach(), labels)

        avg_loss = total_loss / len(data_loader)
        epoch_metrics = self.epoch_metrics.compute_epoch_metrics()
        if self.accelerator.is_local_main_process:
            tqdm.write(f"Eval Loss: {avg_loss:.4f}, Eval Metrics: {epoch_metrics}")
        self.epoch_metrics.reset()

        return avg_loss, epoch_metrics


     



    def validate(self):
        avg_loss, epoch_metrics = self.run_evaluation(self.val_loader)  # This should correctly unpack the tuple
        if self.accelerator.is_local_main_process:
            print(f"Validation Loss: {avg_loss:.4f}")  # This line should now work without error
        return avg_loss, epoch_metrics


    def test(self):
        avg_loss, epoch_metrics = self.run_evaluation(self.test_loader)  # Correctly unpack the tuple
        if self.accelerator.is_local_main_process:
            print(f"Test Loss: {avg_loss:.4f}")
            # Optionally print or log the evaluation metrics
            print(f"Test Metrics: {epoch_metrics}")
        return avg_loss, epoch_metrics  # Optionally return both values if needed elsewhere


    def prepare_data(self, sample):
        # Assuming sample is a dictionary with keys 'text', 'vision', 'audio', 'label'
        input_ids, attention_masks = sample['text']  # Here, 'text' returns a tuple (input_ids, attention_masks)
        vision_data = sample['vision']
        audio_data = sample['audio']
        labels = sample['label']
        
        # Assuming accelerator.prepare() can handle these inputs directly
        prepared_input_ids, prepared_attention_masks, prepared_vision_data, prepared_audio_data, prepared_labels = \
            self.accelerator.prepare(input_ids, attention_masks, vision_data, audio_data, labels)

        return prepared_input_ids, prepared_attention_masks, prepared_vision_data, prepared_audio_data, prepared_labels


    def train(self, epochs: int):
        for epoch in range(epochs):
            if self.accelerator.is_local_main_process:
                print(f"Epoch {epoch+1}/{epochs}")
            self.train_epoch()
            self.validate()
        self.test()
        self.metrics.reset()
        if self.accelerator.is_local_main_process:
            print("Training completed. Here are the final metrics:")
            print(self.metrics.compute())
            
    def save_model(self, save_path):
    
        model_state_dict = self.model.state_dict()
        torch.save(model_state_dict, save_path)
        if self.accelerator.is_local_main_process:
            print(f"Model saved to {save_path}")

        



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
# Invert the unified label mapping
tensor_dataset = torch.load(r"E:\text_datasets\saved\train_emotion_no_batch_no_batch.pt")
# Adjust the unpacking based on the actual structure of your TensorDataset




from torch.utils.data import random_split
dataset_path = r"F:\FP_multimodal\MELD\MELD_RAW\train\MELD_train"
dataset = MultimodalMELDDataset(data_dir=dataset_path)
# Define the proportions for your splits
train_size = int(0.7 * len(dataset))  # 70% of the dataset for training
val_size = int(0.15 * len(dataset))  # 15% for validation
test_size = len(dataset) - train_size - val_size  # Remaining 15% for testing
# Just for debugging, right before the training loop starts
  # or print(type(sample)) and print(sample.keys()) for more structured output
    
# Set the seed for reproducibility
torch.manual_seed(42)

# Perform the split
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
print(len(train_dataset))
# Now you can pass these datasets to your classifier
classifier = MultimodalClassifier(train_dataset=train_dataset,
                                  val_dataset=val_dataset,
                                  test_dataset=test_dataset,
                                  num_classes=9,
                                  text_state_dict=text_dict,
                                  vision_state_dict=vision_dict,
                                  audio_state_dict=speech_dict,
                                  audio_label_map=speech_to_unified,
                                  text_label_map=text_label_mapping,
                                  vision_label_map=vision_to_unified,
                                  unified_label_map = unified_label_mapping)

classifier.train(epochs=10)
classifier.save_model("EmoHydra-10e-170p-MELD")
