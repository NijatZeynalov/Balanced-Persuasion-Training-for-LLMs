from .dialogue_simulator import DialogueSimulator
from .evaluation import PersuasionEvaluator
from .dataset import PersuasionDataset  # Import the dataset class
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F


class PersuasionTrainer:
    """
    Handles the training of the persuasion model.
    """

    def __init__(self, config, model_instance):
        self.config = config
        self.model = model_instance
        self.simulator = DialogueSimulator(model_instance, config)
        self.evaluator = PersuasionEvaluator(config, self.simulator)
        self.optimizer = Adam(self.model.model.parameters(), lr=self.config.learning_rate)

    def train(self):
        """
        Train the persuasion model.
        """
        # Load dataset from user-provided path
        dataset = PersuasionDataset(self.config)
        data_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        # Training loop
        print("Training process started...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.model.to(device)
        self.model.model.train()
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            for batch in data_loader:
                inputs, targets = batch
                inputs = {key: val.to(device) for key, val in inputs.items()}
                targets = targets.to(device)

                # Forward pass
                outputs = self.model.model(**inputs)
                logits = outputs.logits

                # Calculate loss (cross-entropy for language modeling)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            average_loss = epoch_loss / len(data_loader)
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}, Loss: {average_loss:.4f}")

        print("Training complete.")
