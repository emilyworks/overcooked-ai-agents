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

    def __init__(self):
        
        self.env = OverCookedEnv(scenario="asymmetric_advantages")
        self.initial = self.env.reset()
        self.tree = Tree(Node(self.initial, None, [[-1,-1]], [-1,-1], 0, 1, 0))

    def iterate(self, n):
        '''
        Perform MCTS n times to develop the visit frequencies that will be later used by path function
        '''        

        for _ in range(n):

            state, reward = self.selection(self.initial)

            if reward <= 0:
                expanded = self.expansion(state, False)
                reward = self.rollout(expanded)
                self.backprop(expanded, reward)

            else:
                expanded = self.expansion(state, True)
                reward = self.rollout(expanded)
                self.backprop(expanded, reward)

            self.env.reset()
            env.reset()


    
    def path(self):
        '''
        Determine the chosen path of agent actions based on MCTS algo results
        '''
        #initializations
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

        return mcts_path 

    def selection(self, state, alpha=0.01):
        '''
        selects a node in the tree that has not been expanded yet
        '''
        while self.tree.get(state) != None:
            
            #get the children
            children = self.children(state)
        
            #initialize values with one of kids - e.g. first
            first_kiddo = children[0]

            #if child is not in the tree yet...
            if self.tree.get(first_kiddo.state) == None:
                return state, self.tree.get(state).reward
            
            #otherwise, continue initialization
            selected = first_kiddo
            selected_reward = first_kiddo.reward
            
            selected_uct = first_kiddo.wins/first_kiddo.visits + alpha*(np.sqrt((np.log(first_kiddo.parent.visits))/first_kiddo.visits))

            #now compare to the rest of the children
            for child in children[1:]:
                
                #if we've got a kid not in the tree...
                if self.tree.get(child.state) == None:
                    return state, self.tree.get(state).reward
                
                #if the child is is in tree...compare uct values to find the best kiddo and then loop again
                else:
                    uct_val = (child.wins)/(child.visits) + alpha*(np.sqrt((np.log(child.parent.visits))/child.visits))
                    if uct_val > selected_uct:
                        selected_uct = uct_val
                        selected = child
                        selected_reward = child.reward
            
            #update values for the next iteration
            state = selected.state
        
        return state, self.tree.get(state).reward

    def expansion(self, state, done):
        '''
        expand/add to the tree a child node from the selected node
        '''
        #if the state is terminal, there is nothing to expand. just return state
        if self.tree.get(state).reward > 0:
            return state
        
        #otherwise...
        else:
            children = self.children(state)

            for child in children:

                if self.tree.get(child.state) == None:
                    self.tree.add(Node(child.state, child.parent, child.action_path, child.action_from_parent, child.wins, child.visits, child.reward ))
                    return child.state

        #if all the children are already in the tree (though that shouldn't happen)...
        return state   
    
    
    def rollout(self, state):
        '''
        implement random rollout policy frm expanded node and return reward from policy
        '''
        reward = 0
        done = False

        #while we are not in a terminal state and the gameplay session has not yet maxed out all turns...
        while reward <= 0 and not done:
            
            #get kiddos
            # children = self.children(state)

            #sample a random action
            action = np.array([self.env.action_space.sample() for _ in range(2)])  

            #update the state 
            next_state, reward, done, info = self.env.step(action=action)
            state = next_state

            # done += 1
        
        return reward


    def backprop(self, state, reward):
        '''
        Takes in the terminal state and updates the wins and visits for the entire tree path
        '''
        current = self.tree.get(state)

        while current != None:

            #increment wins
            if reward > 0:
                current.wins += 1

            #increment the visits
            current.visits += 1

            #update current
            current = current.parent
    
    def children(self, state):
        '''
        given a state, returns all its children states and their associated actions as nodes
        '''

        #create shadow environment to simulate movement through the tree 
        shadow_env = OverCookedEnv(scenario='asymmetric_advantages')

        current = self.tree.get(state)
        path = []
        
        #get the action path for the state by traversing back to root
        while current != None:
            path.append(current.action_from_parent)
            current = current.parent

        path.reverse()

        #children actions
        child_actions = [0, 1, 2, 3, 4, 5]
        children = []

        #append in the children of that state
        for action1 in child_actions:
            for action2 in child_actions:
                
                #get to the state first from init
                shadow_env.reset()
                for action in path:
                    shadow_env.step(action=action)
                
                #from the state, then take the action corresponding to child 
                joint_action = [action1, action2]
                child_state, reward, done, info = shadow_env.step(action=joint_action)

                child = Node(child_state, self.tree.get(state), path + [joint_action], joint_action, 0, 1, reward)
                children.append(child)

        shadow_env.reset()

        return children

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
    
    

