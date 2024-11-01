class DialogueTree:
    """
    Generates a dialogue tree based on the simulated conversations.
    """
    def __init__(self, initial_prompt, is_positive_persuasion):
        self.root = DialogueNode(initial_prompt, is_positive_persuasion)
        self.current_node = self.root

    def add_dialogue_step(self, persuader_response, persuadee_response):
        """
        Add a new dialogue step to the tree.
        """
        new_node = DialogueNode(persuadee_response, self.current_node.is_positive_persuasion)
        self.current_node.add_child(new_node, persuader_response)
        self.current_node = new_node

    def get_dialogue_tree(self):
        """
        Get the complete dialogue tree.
        """
        return self.root

class DialogueNode:
    """
    Represents a node in the dialogue tree.
    """
    def __init__(self, text, is_positive_persuasion):
        self.text = text
        self.is_positive_persuasion = is_positive_persuasion
        self.children = []
        self.persuader_responses = []

    def add_child(self, child_node, persuader_response):
        """
        Add a child node to the current node.
        """
        self.children.append(child_node)
        self.persuader_responses.append(persuader_response)