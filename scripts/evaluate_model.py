from persuasion_trainer.config import Config
from persuasion_trainer.persuasion_model import PersuasionModel
from persuasion_trainer.evaluation import PersuasionEvaluator
from persuasion_trainer.dialogue_simulator import DialogueSimulator

def main():
    config = Config()
    persuadee_model = PersuasionModel(config)
    persuadee_model.load_model()

    dialogue_simulator = DialogueSimulator(persuadee_model, config)
    persuasion_evaluator = PersuasionEvaluator(config, dialogue_simulator)

    # Evaluate the model's performance
    persuasion_evaluator.evaluate_model(persuadee_model)

if __name__ == "__main__":
    main()
