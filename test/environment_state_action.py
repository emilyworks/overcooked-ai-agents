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
from agents import Random, MCTS


#initialize the environment and the state
env = OverCookedEnv(scenario="asymmetric_advantages")
env.reset()

#hello
# INSTRUCTIONS FOR BELOW: "uncomment" one agent at a time 

# Random agent gameplay ---------------------------------------UNCOMMENT THE SECTION BELOW TO RUN THE AGENTS

# for _ in range(200):

#     #call the class and its methods
#     agent = Random(env)
#     agent_ac1 = agent.action1()
#     agent_ac2 = agent.action2()
    
#     #feed actions into the environment
#     next_state, reward, done, info = env.step(action=[agent_ac1, agent_ac2])
    
#     #visualize the environment
#     image = env.render()
#     cv2.imshow('Image', image)
#     key = cv2.waitKey(100)

# MCTS agent gameplay -------------------------------------------UNCOMMENT THE SECTION BELOW TO RUN THE AGENTS

# agent = MCTS()
# agent_action = agent.iterate(1)
# agent_path = agent.path()

# print(agent_path)

# for action in agent_path:
#     next_state, reward, done, info = env.step(action=action)
#     image = env.render()
#     cv2.imshow('Image', image)
#     key = cv2.waitKey(100)


# ARCHIVED MATERIAL (DEPRECATED) -----------------------------------------------------------

    #     # get the possibe list of actions
    #     action = np.array([env.action_space.sample() for _ in range(2)])
    #     # print(path)
    #     # action[0] = action
    #     action[1] = 4
    #     # input_action = 4
    # '''(Old) Random Agent-------------------------------------------------------------------------------------'''
    #keep track of the rewards
    # sum_rewards += reward

    #select a random action
    # input_action = random.randint(0,5)

    # steppy.append(_)
    # sum_sum.append(sum_rewards)


    # ''' Print State '''

    # onehot_state = env.get_onehot_state()[0]
    # print(onehot_state.shape)
    # width, height, channel = onehot_state.shape[1], onehot_state.shape[0], onehot_state.shape[2]
    # print(width, height, channel)
    # count_width = 6
    # count_height = round(channel / count_width)

    # axes = []
    # fig = plt.figure()

    # for a in range(count_height * count_width):
    #     b = onehot_state[:, :, a]
    #     b = cv2.rotate(b, 2)
    #     b = cv2.flip(b, 0)
    #     axes.append(fig.add_subplot(count_height, count_width, a + 1))
    #     subplot_title = (str(a))
    #     axes[-1].set_title(subplot_title)
    #     plt.imshow(b)
    # fig.tight_layout()
    # plt.show()

    # ''' Input Action ''' --- THIS IS FOR MANUAL KEYBOARD MANIPULATION OF THE AGENT'S STEPS 
    # sum_rewards += reward
    # input_action = random.randint(0,5)
    # if key == ord('a'):
    #     input_action = 3
    # elif key == ord('s'):
    #     input_action = 1
    # elif key == ord('d'):
    #     input_action = 2
    # elif key == ord('w'):
    #     input_action = 0
    # elif key == ord('m'):
    #     input_action = 5
    # else:
    #     input_action = 4
    # steppy.append(_)
    # sum_sum.append(sum_rewards)
    # if loop == 1000:
    #     print("==========================================")
    #     print(loop)
    #     # print(steppy)
    #     print(sum(sum_sum))
    #     break
    # if done:
    #     steppy.append(_)
    #     sum_sum.append(sum_rewards)
    #     if loop == 1000:
    #         print("==========================================")
    #         print("done")
    #         break
    #     state = env.reset()
    # else:
    #     state = next_state

# print(sum_sum)
# print(steppy)

# for _ in range(10000):
#     action = np.array([env.action_space.sample() for _ in range(2)])
#     action[0] = input_action
#     action[1] = 4
#     input_action = 4
#     next_state, reward, done, info = env.step(action=action)

#     image = env.render()
#     cv2.imshow('Image', image)
#     print(reward, done)
#     key = cv2.waitKey(0)

#     ''' Print State '''
#     onehot_state = env.get_onehot_state()[0]
#     print(onehot_state.shape)
#     width, height, channel = onehot_state.shape[1], onehot_state.shape[0], onehot_state.shape[2]
#     print(width, height, channel)
#     count_width = 6
#     count_height = round(channel / count_width)

#     axes = []
#     fig = plt.figure()

#     for a in range(count_height * count_width):
#         b = onehot_state[:, :, a]
#         b = cv2.rotate(b, 2)
#         b = cv2.flip(b, 0)
#         axes.append(fig.add_subplot(count_height, count_width, a + 1))
#         subplot_title = (str(a))
#         axes[-1].set_title(subplot_title)
#         plt.imshow(b)
#     fig.tight_layout()
#     plt.show()

#     ''' Input Action '''
#     if key == ord('a'):
#         input_action = 3
#     elif key == ord('s'):
#         input_action = 1
#     elif key == ord('d'):
#         input_action = 2
#     elif key == ord('w'):
#         input_action = 0
#     elif key == ord('m'):
#         input_action = 5
#     else:
#         input_action = 4

#     if done:
#         state = env.reset()
#     else:
#         state = next_state

