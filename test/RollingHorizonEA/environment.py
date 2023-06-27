"""
Class that needs to be inherited in any environment for the Rolling Horizon Evolutionary Algorithm to work
"""
import cv2
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import overcooked_ai_py
from overcooked_ai_py.env import OverCookedEnv
# from search.py import SearchTree
import random
import numpy.typing as npt
from dataclasses import dataclass
from RollingHorizonEA import RollingHorizonEvolutionaryAlgorithm
from RollingHorizonEA.environment import Environment
#initialize the game environment and the state
env = OverCookedEnv(scenario="asymmetric_advantages")
state = env.reset()
class Environment():

    def __init__(self):
        self.env = OverCookedEnv(scenario="asymmetric_advantages")
        self.initial = self.env.reset()
        self.tree = Tree(Node(self.initial, None, [[-1, -1]], [-1, -1], 0, 1, 0))
        self.sum_rewards = 0

    def perform_action(self, n):
        # refer to Emily's MCTS code
        for _ in range(n):
            state, reward = self.selection(self.initial)
            if reward <= 0:
                expanded = self.expansion(state, False)
                reward = self.evaulute_rollout(expanded)
                self.is_game_over(expanded, reward)
            else:
                expanded = self.expansion(state, True)
                reward = self.rollout(expanded)
                self.is_game_over(expanded, reward)

            self.env.reset()
            env.reset()

    def evaluate_rollout(self, state):
        reward = 0
        done = False
        # while we are not in a terminal state and the gameplay session has not yet maxed out all turns...
        while reward <= 0 and not done:
            # get kiddos
            # children = self.children(state)
            # sample a random action
            action = np.array([self.env.action_space.sample() for _ in range(2)])
            # update the state
            next_state, reward, done, info = self.env.step(action=action)
            state = next_state
            # done += 1
        return reward

    def path(self):
        '''
        Determine the chosen path of agent actions based on MCTS algo results
        '''
        # initializations
        most_visits = 0
        done = 0
        reward = 0
        current = self.initial
        mcts_path = []

        while done <= 200 and reward <= 0:

            children = self.children(current)
            for child in children:
                child = self.tree.get(child.state)
                if child == None:
                    continue
                if child.reward > 0:
                    mcts_path.append(child.action_from_parent)
                    return mcts_path
                if child.visits > most_visits:
                    most_visits = child.visits
                    current = child.state
            mcts_path.append(self.tree.get(current).action_from_parent)
            done += 1

        return rhea_path

    def get_random_action(self):
        # refer to Emily's MCTS code
        #pull it from the tree
        #currenlty used the random action function
        return random.randint(0, 5)

    def is_game_over(self):
        return True

    def get_current_score(self):
        # refer to Emily's MCTS code
        self.sum_rewards += 1
        return self.sum_rewards

    def ignore_frame(self):
        #we dont know what this is for
        raise NotImplementedError


@dataclass
class Node:
    state: npt.ArrayLike
    parent: "Node"
    action_path: list[list[int]]
    action_from_parent: list[int]
    wins: int
    visits: int
    reward: int


class Tree:
    def __init__(self, root: "Node"):
        self.nodes = {root.state.tobytes(): root}

    def add(self, node: "Node"):
        self.nodes[node.state.tobytes()] = node

    def get(self, state: npt.ArrayLike):
        flat_state = state.tobytes()
        if flat_state not in self.nodes:
            return None
        return self.nodes[flat_state]


if __name__ == "__main__":
    num_dims = 600
    num_evals = 50
    rollout_length = 100
    mutation_probability = 0.1

    # Set up the problem domain as one-max problem
    environment = Environment()
    #Environment().perform_action(1)
    # agent_path = Environment.path(self)
    rhea = RollingHorizonEvolutionaryAlgorithm(rollout_length, environment, mutation_probability, num_evals)
    rhea.run()
    # for action in agent_path:
    #      next_state, reward, done, info = env.step(action=action)
    #      image = env.render()
    #      cv2.imshow('Image', image)
    #      key = cv2.waitKey(100)

    # agent = MCTS()
    # agent_action = agent.iterate(1)
    # agent_path = agent.path()

    # print(agent_path)

    # for action in agent_path:
    #     next_state, reward, done, info = env.step(action=action)
    #     image = env.render()
    #     cv2.imshow('Image', image)
    #     key = cv2.waitKey(100)
