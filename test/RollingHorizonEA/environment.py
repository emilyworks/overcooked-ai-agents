"""
Class that needs to be inherited in any environment for the Rolling Horizon Evolutionary Algorithm to work
"""


#initialize the game environment and the state
env = OverCookedEnv(scenario="asymmetric_advantages")
state = env.reset()
class Environment():

    def __init__(self):
        self.env = OverCookedEnv(scenario="asymmetric_advantages")
        self.initial = self.env.reset()
        self.tree = Tree(Node(self.initial, None, [[-1, -1]], [-1, -1], 0, 1, 0))

    def perform_action(self, action):
        # refer to Emily's MCTS code
        # we need to implement the expansion function and
        #this is the iterate function from Emilys MCTS code
        for _ in range(n):

            state, reward = self.selection(self.initial)

            if reward <= 0:
                expanded = self.expansion(state, False)
                reward = self.rollout(expanded)
                self.is_game_over(expanded, reward)


            else:
                expanded = self.expansion(state, True)
                reward = self.rollout(expanded)
                self.backprop(expanded, reward)

            self.env.reset()
            env.reset()

    def evaluate_rollout(self, solution, discount_factor=0, ignore_frames=0):
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


    def get_random_action(self):
        # refer to Emily's MCTS code
        #pull it from the tree
        #currenlty used the random action function
        return random.randint(0, 5)

    def is_game_over(self):
        # refer to Emily's MCTS code
        #Used Emilys backprop function
        current = self.tree.get(state)
        while current != None:
            # increment wins
            if reward > 0:
                current.wins += 1
            # increment the visits
            current.visits += 1
            # update current
            current = current.parent

    def get_current_score(self):
        # refer to Emily's MCTS code
        return current.wins

    def ignore_frame(self):
        #we dont know what this is for
        raise NotImplementedError
