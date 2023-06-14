"""
Class that needs to be inherited in any environment for the Rolling Horizon Evolutionary Algorithm to work
"""


class Environment():

    def __init__(self, name):
        self._name = name

    def perform_action(self, action):
        # refer to Emily's MCTS code
        raise NotImplementedError

    def evaluate_rollout(self, solution, discount_factor=0, ignore_frames=0):
        # refer to Emily's MCTS code
        raise NotImplementedError

    def get_random_action(self):
        # refer to Emily's MCTS code
        raise NotImplementedError

    def is_game_over(self):
        # refer to Emily's MCTS code
        raise NotImplementedError

    def get_current_score(self):
        # refer to Emily's MCTS code
        raise NotImplementedError

    def ignore_frame(self):
        raise NotImplementedError
