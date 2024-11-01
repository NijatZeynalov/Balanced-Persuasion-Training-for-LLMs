import torch
import pandas as pd
from torch.utils.data import Dataset

class PersuasionDataset(Dataset):
    """
    A custom Dataset class to load training data for the persuasion model.
    Supports both CSV and JSON formats.
    """
    def __init__(self, config):
        self.config = config
        self.data = self._load_data()

    def _load_data(self):
        """
        Load training data from user-provided file (CSV or JSON).
        """
        if self.config.dataset_format == "csv":
            df = pd.read_csv(self.config.dataset_path)
        elif self.config.dataset_format == "json":
            df = pd.read_json(self.config.dataset_path)
        else:
            raise ValueError("Unsupported dataset format. Use 'csv' or 'json'.")

        # Convert dialogue pairs to tokenized inputs and targets
        data = []
        for _, row in df.iterrows():
            initial_prompt = row["initial_prompt"]
            response = row["response"]

            # Tokenize the dialogue and create training data
            inputs = self.config.tokenizer(initial_prompt, return_tensors="pt", padding=True, truncation=True)
            targets = self.config.tokenizer(response, return_tensors="pt", padding=True, truncation=True)["input_ids"]

            data.append((inputs, targets))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
