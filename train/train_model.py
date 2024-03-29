import torch
from torch.utils.data import DataLoader, Dataset
from FallingPlanet.orbit.models.multimodal.Hydra import HydraTiny
from FallingPlanet.orbit.models.multimodal.Chimera import Chimera
from FallingPlanet.orbit.models.AuTransformer import FPATF_Tiny
from FallingPlanet.orbit.models.BertFineTuneForSequenceClassification import BertFineTuneTiny
from FallingPlanet.orbit.models.DeiTFineTuneForImageClassification import DeitFineTuneTiny
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from accelerate import Accelerator
from FallingPlanet.orbit.utils.model_utils import MetricsWrapper

class MultimodalClassifier:
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset, num_classes: int, batch_size: int = 32, learning_rate: float = 0.001, text_state_dict: str = "", vision_state_dict: str = "", audio_state_dict: str = "", mode: str = 'macro', vision_label_map: dict = None, text_label_map: dict = None, audio_label_map: dict = None):
        self.accelerator = Accelerator()

        # Initialize the model with state dictionaries for each modality
        self.model = HydraTiny(num_classes=num_classes, feature_dim=128, requires_grad=False,text_label_map=text_label_map,audio_label_map=audio_label_map,vision_label_map=vision_label_map)
        self.model.load_modal_state_dicts(text_dict=text_state_dict, audio_dict=audio_state_dict, vision_dict=vision_state_dict)
        self.loss_function = CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.model, self.optimizer, self.train_loader, self.val_loader, self.test_loader = self.accelerator.prepare(
        self.model, self.optimizer, self.train_loader, self.val_loader, self.test_loader)

        self.metrics = MetricsWrapper(num_classes=num_classes, device=self.accelerator.device, mode=mode)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for sample in tqdm(self.train_loader, desc="Training", disable=not self.accelerator.is_local_main_process):
            text_data, vision_data, audio_data, labels = self.prepare_data(sample)

            self.optimizer.zero_grad()
            outputs, vision_labels, text_labels, audio_labels = self.model(vision_data, text_data, audio_data)
            loss = self.loss_function(outputs, labels)
            self.accelerator.backward(loss)
            self.optimizer.step()

            running_loss += loss.item()

            self.metrics.update(outputs, labels)

        avg_loss = running_loss / len(self.train_loader)
        if self.accelerator.is_local_main_process:
            print(f"Training Loss: {avg_loss:.4f}")
        return avg_loss

    def run_evaluation(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for sample in tqdm(data_loader, desc="Evaluating", disable=not self.accelerator.is_local_main_process):
                text_data, vision_data, audio_data, labels = self.prepare_data(sample)
                outputs, vision_labels, text_labels, audio_labels = self.model(vision_data, text_data, audio_data)
                loss = self.loss_function(outputs, labels)
                total_loss += loss.item()

                self.metrics.update(outputs, labels)

        avg_loss = total_loss / len(data_loader)
        return avg_loss

    def validate(self):
        avg_loss = self.run_evaluation(self.val_loader)
        if self.accelerator.is_local_main_process:
            print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def test(self):
        avg_loss = self.run_evaluation(self.test_loader)
        if self.accelerator.is_local_main_process:
            print(f"Test Loss: {avg_loss:.4f}")
        return avg_loss

    def prepare_data(self, sample):
        text_data, vision_data, audio_data, labels = sample['text'], sample['vision'], sample['audio'], sample['label']
        return self.accelerator.prepare(text_data, vision_data, audio_data, labels)

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


# Example usage remains largely the same, but now it's prepared for distributed training
unified_label_mapping = {
    "anger": 0,
    "angry": 0,
    "disgust": 1,  # Assuming "disgust" is added here if we're following alphabetical order
    "fear": 2,
    "joy": 3,
    "love": 4,
    "neutral": 5,
    "sadness": 6,
    "surprise": 7,
    "worry": 8  # "worry" might move to 8 if "disgust" is inserted before it
}
speech_dict = r"E:\model_saves\EmoSpeak_Transformer_Tinier.pt"
vision_dict = r"D:\Users\WillR\Documents\GitHub\EmoVision\EmoVision_augmented-tiny.pth"
text_dict = r"D:\Users\WillR\Documents\GitHub\EmoBERTv2\EmoBERTv2-tiny.pth"
# Adjusted unified label mapping to include all emotions
unified_label_mapping = {
    "anger": 3,
    "angry": 3,
    "disgust": 5,
    "fear": 4,
    "joy": 1,
    "happy": 1,
    "love": 4,  # Assuming "love" can be treated uniquely or mapped closely to positive emotions like "joy"
    "neutral": 0,
    "sadness": 2,
    "sad": 2,
    "surprise": 6,
    "worry": 8  # Assigning "worry" a unique value as it represents a distinct emotion
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

# Vision model array to unified label mapping, including "love" and "worry"
vision_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise", "love", "worry"]
vision_to_unified = {label: unified_label_mapping[label] for label in vision_labels}


from torch.utils.data import random_split
dataset = r"F:\FP_multimodal\MELD\MELD-RAW\train\MELD_train"
# Define the proportions for your splits
train_size = int(0.7 * len(dataset))  # 70% of the dataset for training
val_size = int(0.15 * len(dataset))  # 15% for validation
test_size = len(dataset) - train_size - val_size  # Remaining 15% for testing

# Set the seed for reproducibility
torch.manual_seed(42)

# Perform the split
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Now you can pass these datasets to your classifier
classifier = MultimodalClassifier(train_dataset=train_dataset,
                                  val_dataset=val_dataset,
                                  test_dataset=test_dataset,
                                  num_classes=9,
                                  text_state_dict=text_dict,
                                  vision_state_dict=vision_dict,
                                  audio_state_dict=speech_dict,
                                  audio_label_map=speech_to_unified,
                                  text_label_map=unified_label_mapping,
                                  vision_label_map=vision_to_unified)

