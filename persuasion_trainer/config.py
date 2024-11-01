class Config:
    """
    Configuration settings for the Persuasion-Balanced Training framework.
    """

    def __init__(self):
        self.model_name = "decapoda-research/llama-2-7b"
        self.max_dialogue_length = 10
        self.max_tree_depth = 5
        self.positive_persuasion_ratio = 0.5
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.num_epochs = 10
        self.output_dir = "output/"
        self.dataset_path = "data/persuasive_dialogues.csv"
        self.dataset_format = "csv"
