from .dialogue_tree import DialogueTree

class DialogueSimulator:
    """
    Simulates dialogues based on the persuasion model.
    """
    def __init__(self, model_instance, config):
        self.model = model_instance
        self.config = config

    def simulate_dialogue(self, initial_prompt, is_positive_persuasion):
        """
        Simulate a dialogue using the persuasion model.
        """
        dialogue_tree = DialogueTree(initial_prompt, is_positive_persuasion)
        current_node = dialogue_tree.root
        dialogue_steps = [initial_prompt]

        for _ in range(self.config.max_dialogue_length):
            response = self.model.predict(current_node.text)
            dialogue_steps.append(response)
            current_node = current_node.add_child(response)

        return dialogue_tree, dialogue_steps
