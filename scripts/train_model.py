from persuasion_trainer.config import Config
from persuasion_trainer.trainer import PersuasionTrainer
from persuasion_trainer.persuasion_model import PersuasionModel

def main():
    config = Config()
    config.dataset_path = "data/custom_persuasive_dialogues.csv"
    config.dataset_format = "csv"

    persuadee_model = PersuasionModel(config)
    persuadee_model.load_model()

    trainer = PersuasionTrainer(config, persuadee_model)
    trainer.train()

if __name__ == "__main__":
    main()
