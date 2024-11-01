# Balanced Persuasion Training for LLMs

## Overview

This project aims to train a language model to effectively engage in persuasive conversations. 

The framework provides tools for training, simulating dialogues, and evaluating the persuasion quality of generated conversations.

## Features

* Train a language model for persuasive conversations.

* Simulate dialogues to improve the model's persuasive skills.

* Evaluate the effectiveness of the trained model's persuasive techniques.

* Allow users to provide their own dataset of persuasive conversations.

## Usage

### Preparing Your Dataset

Prepare your dataset as a CSV or JSON file.

The dataset should have at least two columns: initial_prompt and response.

Update the configuration in config.py with your dataset path and format:

```
self.dataset_path = "data/your_dataset.csv"
self.dataset_format = "csv" 
```
### Training the Model

Edit the train_model.py script to point to your dataset if needed.

Run the training script:
```
python scripts/train_model.py
```
This will start training the model based on your dataset. The model and tokenizer will be loaded and saved to the specified output directory.

The model can be replaced with other language models available from transformers. Update model_name in config.py to use a different model. Default is Llama 2 7b.
