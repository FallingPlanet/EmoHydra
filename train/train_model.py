import torch
from torch.utils.data import DataLoader, Dataset
from FallingPlanet.orbit.models.multimodal.Hydra import HydraTiny
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

class MultimodalClassifier:
    def __init__(self, dataset: Dataset, num_classes: int, batch_size: int = 32, learning_rate: float = 0.001):
        self.model = HydraTiny(num_classes=num_classes)
        self.loss_function = CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for sample in tqdm(self.data_loader, desc="Training"):
            text_data, vision_data, audio_data, labels = self.prepare_data(sample)
            
            self.optimizer.zero_grad()
            outputs = self.model(text_data, vision_data, audio_data)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        print(f"Training Loss: {running_loss / len(self.data_loader):.4f}")

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        # Placeholder for validation logic
        print("Validation Placeholder")

    def test(self):
        self.model.eval()
        running_loss = 0.0
        # Placeholder for testing logic
        print("Testing Placeholder")

    def prepare_data(self, sample):
        text_data = sample['text'].to(self.device)
        # Assuming vision data is a list of tensors; no need for torch.stack
        vision_data = [frame.to(self.device) for frame in sample['vision']]
        audio_data = sample['audio'].to(self.device)
        labels = sample['label'].to(self.device)
        return text_data, vision_data, audio_data, labels

    def train(self, epochs: int):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            self.train_epoch()
            self.validate()  # Call to validation loop
        self.test()  # Final testing after all epochs

# Example usage:
# Assuming unified_label_mapping and dataset are already defined
classifier = MultimodalClassifier(dataset=dataset, num_classes=len(unified_label_mapping))
classifier.train(epochs=10)
