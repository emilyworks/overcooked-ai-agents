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


#initialize the game environment and the state
env = OverCookedEnv(scenario="asymmetric_advantages")
state = env.reset()

#Random Agent ------------------------------------------
'''
This class specifies the joint actions for two agents following a random gameplay policy. A
Actions can be represented by integers of the range [0,5] inclusive. 
'''

class Random():

    def __init__(self, env):
        self.sum_rewards = 0
        self.env = env
    
    def action1(self):
        '''
        returns action for the first agent
        '''
        return random.randint(0,5)
    
    def action2(self):
        '''
        returns action for the second agent
        '''
        return random.randint(0,5)
    
    def reward(self, action):
        self.sum_rewards += self.env.step(action=action)[1]
        return self.sum_rewards


#MCTS Agent ----------------------------------------------------------------------
'''
Below is a simple version of a cooperative MCTS joint action. The algorithm runs through n iterations of
selection, expansion, rollout, and backpropagation to develop a "tree" containing information about states
and the frequency of visits to those states. The "path" method then selects the sequence of actions -- corresponding 
to sequence of joint gameplay actions -- that contain the highest visit frequencies.

CREDIT: the Node and Tree utility classes used below the MCTS class are adapted from the Spring 2023 Class
of "Artificial Intelligence".

'''

class MCTS():

    def __init__(self, initial, env):
        
        self.tree = Tree(Node(initial, None, True, None, 0, 1, 0, False))
        self.env = env
        self.initial = initial

    def iterate(self, n):
        '''
        Perform MCTS n times to develop the visit frequencies that will be later used by path function
        '''        

        for _ in range(n):
            state, reward = self.selection(self.initial)
            if reward == 0:
                expanded = self.expansion(state, False)
                reward = self.rollout(expanded, False, 0)
                self.backprop(expanded)
            else:
                expanded = self.expansion(state, True)
                reward = self.rollout(expanded, True, reward)
                self.backprop(expanded)
            self.env.reset()
            env.reset()
    
    def path(self):
        '''
        Determine the chosen path of agent actions based on MCTS algo results
        '''
        #reset env
        env.reset()
        self.env.reset()

        most_visits = 0
        done = False
        current = self.tree.get(self.initial)

        #path list
        action_path = []
        chosen_action = current.action_from_parent
        

        children_actions = [0, 1, 2, 3, 4, 5]
        
        while not done:
            
            for child_action1 in children_actions:
                for child_action2 in children_actions:

                    try:
                        child_state, child_reward, done = self.env.step(action=[child_action1, child_action2])[:3]
                    except:
                        continue

                    else:
                        # for node in self.tree:
                        # print(self.tree)
                        if self.tree.get(child_state) != None:

                            visits = self.tree.get(child_state).visits

                            if visits > most_visits:
                                most_visits = visits
                                chosen_child = child_state
                                chosen_action = [child_action1, child_action2]
                
            action_path.append(chosen_action)
            most_visits = 0

        print(action_path)

        return action_path 

    def selection(self, state, alpha=0.01):
        '''
        selects a node in the tree that has not been expanded yet
        '''

        children_actions = [0,1,2,3,4,5]
        
        #initialize values with one of the children if possible -- let's say the first kid
        first_kiddo, first_reward, terminal = env.step(action=[children_actions[0], 0])[0:3]

        if self.tree.get(first_kiddo) == None:
            print("HELLO")
            return state, self.tree.get(state).reward
        
        selected = first_kiddo
        selected_reward = first_reward

        wins = self.tree.get(first_kiddo).wins
        visits = self.tree.get(first_kiddo).visits
        parent = self.tree.get(first_kiddo).parent
        parent_visits = parent.visits
        selected_uct = wins/visits + alpha*(np.sqrt((np.log(parent_visits))/visits))
        

        #continue with the rest of the algo -- picking best kids
        #calculate each child's UCT value 
        while self.tree.get(state).terminal != True:
            for child_action1 in children_actions:
                for child_action2 in children_actions:
                    
                    child_state, child_reward, terminal = env.step(action=[child_action1, child_action2])[:3]
                    print("THIS IS MY CHILD STATE")
                    print(child_state)

                    # if child_state.all() == state.all():
                    #     continue

                    #if child not in state...
                    if self.tree.get(child_state) == None: 
                        print("WHYYYY")
                        return state, self.tree.get(state).reward
                    else:
                        #calc UCT value
                        wins = self.tree.get(child_state).wins
                        visits = self.tree.get(child_state).visits
                        parent = self.tree.get(child_state).parent
                        parent_visits = parent.visits
                        uct = wins/visits + alpha*(np.sqrt((np.log(parent_visits))/visits))

                        if uct >= selected_uct:
                            selected_uct = uct
                            selected = child_state
                            selected_reward = child_reward
                            selected_terminal = terminal
                            
            #update the values for the next iteration
            state = selected
            state_reward = selected_reward
            terminal = selected_terminal

        #if state is terminal, just return the state
        return state, self.tree.get(state).reward

    def expansion(self, state, done):
        '''
        expand/add to the tree a child node from the selected node
        '''

        if done:
            return state
        else:
            env.reset()
            children_actions = [0, 1, 2, 3, 4, 5]
            for child_action1 in children_actions:
                for child_action2 in children_actions:
                    child_state, reward, done, blah = env.step(action=[child_action1, child_action2])

                    if self.tree.get(child_state) == None:
                        print("FIRST")

                        new_node = Node(child_state, self.tree.get(state), False, [child_action1, child_action2], 0, 0, reward, done)
                        self.tree.add(new_node)
                        print(self.tree.get(child_state).state)
                        print(child_action1, child_action2)
                        print("======")

                        return new_node.state
                    else:
                        print("SECOND")

        return state 
    
    def rollout(self, state, done, init_reward):
        '''
        implement random rollout policy frm expanded node and return reward from policy
        '''

        while not done:
            action = np.array([self.env.action_space.sample() for _ in range(2)])  
            next_state, reward, done, info = self.env.step(action=action)
            state = next_state
            # print(done)
            # print(reward)

        # print(self.tree.get(state))
        
        return reward


    def backprop(self, state):

        '''
        Takes in the terminal state and updates the wins and visits for the entire tree path
        '''

        current = self.tree.get(state)

        while current != None:

            #increment wins
            if current.reward > 0:
                current.wins += 1

            #increment the visits
            current.visits += 1

            #update current
            current = current.parent

            # print(type(current))

@dataclass
class Node:
    state: npt.ArrayLike
    parent: "Node"
    root: bool 
    action_from_parent: list[int]
    wins: int
    visits: int
    reward: int
    terminal: bool


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

