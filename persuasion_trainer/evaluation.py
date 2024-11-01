import numpy as np
from .dialogue_simulator import DialogueSimulator

class PersuasionEvaluator:
    """
    Evaluates the performance of the persuasion-balanced language model.
    """
    def __init__(self, config, dialogue_simulator: DialogueSimulator):
        self.config = config
        self.dialogue_simulator = dialogue_simulator

    def evaluate_model(self, persuadee_model):
        """
        Evaluate the persuasion-balanced model's performance.
        """
        # Evaluate the model on a set of test dialogues
        test_dialogues = self._generate_test_dialogues()
        persuasion_resistance, persuasion_acceptance = self._evaluate_dialogues(persuadee_model, test_dialogues)

        # Calculate the overall performance metrics
        persuasion_balance = (persuasion_resistance + persuasion_acceptance) / 2
        print(f"Persuasion Resistance: {persuasion_resistance:.2f}")
        print(f"Persuasion Acceptance: {persuasion_acceptance:.2f}")
        print(f"Persuasion Balance: {persuasion_balance:.2f}")

    def _generate_test_dialogues(self):
        """
        Generate a set of test dialogues for evaluation.
        """
        test_dialogues = []
        for _ in range(100):
            initial_prompt = "What is your opinion on renewable energy?"
            is_positive_persuasion = np.random.rand() < 0.5
            _, dialogue_steps = self.dialogue_simulator.simulate_dialogue(initial_prompt, is_positive_persuasion)
            test_dialogues.append((dialogue_steps, is_positive_persuasion))
        return test_dialogues

    def _evaluate_dialogues(self, persuadee_model, test_dialogues):
        """
        Evaluate the test dialogues.
        """
        persuasion_resistance = 0
        persuasion_acceptance = 0
        for dialogue_steps, is_positive_persuasion in test_dialogues:
            # Assume we use the model to predict a score for evaluation purposes
            predicted = persuadee_model.predict(dialogue_steps)
            if is_positive_persuasion:
                persuasion_acceptance += predicted
            else:
                persuasion_resistance += predicted
        return persuasion_resistance / len(test_dialogues), persuasion_acceptance / len(test_dialogues)
