import torch
from torch.utils.data import DataLoader, Dataset
from FallingPlanet.orbit.models.multimodal.Hydra import HydraTiny, HydraTinyRefactored
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
        self.unified_label_map = unified_label_map
        self.num_classes = num_classes
        # Initialize the model with state dictionaries for each modality
        self.model = HydraTinyRefactored(num_classes=num_classes, requires_grad=True,text_label_map=text_label_map,audio_label_map=audio_label_map,vision_label_map=vision_label_map,unified_label_map=unified_label_mapping)
        self.model.load_modal_state_dicts(text_dict=text_state_dict, audio_dict=audio_state_dict, vision_dict=vision_state_dict)
        self.loss_function = CrossEntropyLoss(ignore_index=-1)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.DEFAULT_CLASS_INDEX = num_classes - 1
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=self.custom_collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,collate_fn=self.custom_collate_fn)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,collate_fn=self.custom_collate_fn)

        self.model, self.optimizer, self.train_loader, self.val_loader, self.test_loader = self.accelerator.prepare(
        self.model, self.optimizer, self.train_loader, self.val_loader, self.test_loader)

        self.epoch_metrics = EpochMetricsWrapper(num_classes=num_classes, device=self.accelerator.device, mode=mode)
        

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            
            # Assuming your custom collate function packs each batch correctly
            text_inputs,vision_inputs, audio_inputs, labels = self.prepare_data(batch)
            
            # Forward pass
            outputs = self.model(text_inputs,vision_inputs, audio_inputs)
            
            # Compute loss
            loss = self.loss_function(outputs, labels)
            
            # Backward and optimize
            self.accelerator.backward(loss)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        average_loss = total_loss / len(self.train_loader)
        print(f"Training loss: {average_loss:.4f}")







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

            outputs = self.model(vision_data, text_data, audio_data)
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

        return (prepared_input_ids, prepared_attention_masks), prepared_vision_data, prepared_audio_data, prepared_labels


    def train(self, epochs: int):
        for epoch in range(epochs):
            if self.accelerator.is_local_main_process:
                print(f"Epoch {epoch+1}/{epochs}")
            self.train_epoch()  # Adjusted to no longer require arguments
            avg_loss, epoch_metrics = self.validate()
            print(f"Validation Loss: {avg_loss:.4f}, Validation Metrics: {epoch_metrics}")
        avg_loss, epoch_metrics = self.test()  # This also should be captured
        print(f"Test Loss: {avg_loss:.4f}, Test Metrics: {epoch_metrics}")
        # You may want to reset and display final metrics after the test
        if self.accelerator.is_local_main_process:
            print("Training completed.")
            
    def save_model(self, save_path):
    
        model_state_dict = self.model.state_dict()
        torch.save(model_state_dict, save_path)
        if self.accelerator.is_local_main_process:
            print(f"Model saved to {save_path}")

        



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
                                  audio_label_map=speech_label_mapping,
                                  text_label_map=text_label_mapping,
                                  vision_label_map=vision_label_mapping,
                                  unified_label_map = unified_label_mapping)

classifier.train(epochs=10)
classifier.save_model("EmoHydra-MMA-Attention-MELD")
