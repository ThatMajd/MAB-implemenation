import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# TODO DELTE THIS
from fake_sim import FakeSimulation

import time

EXPLORE = "explore"
EXPLOIT = "exploit"

class Planner:
    def __init__(self, num_rounds, phase_len, num_arms, num_users, arms_thresh, users_distribution):
        """
        :input: the instance parameters (see explanation in MABSimulation constructor)
        """
        # TODO: Decide what/if to store. Could be used in the future
        self.num_rounds = num_rounds
        self.phase_len = phase_len
        self.num_arms = num_arms
        self.num_users = num_users
        self.arms_thresh = arms_thresh.astype(int)
        self.user_distribution = users_distribution

        self.content_ratings = np.zeros((num_users, num_arms))
        self.user_selection = np.zeros(num_rounds)
        self.arm_selection = np.zeros(num_rounds)
        # self.dist_min = np.zeros((num_users, num_arms))
        self.dist_max = np.zeros((num_users, num_arms))
        self.keep_alive = None
        self.arm_pulled = None
        self.reset_keep_alive()
        self.curr_round = -1

        self.norms = []
        self.converged = False

        #self.arms_thresh[0] = 0

    def reset_keep_alive(self):
        self.keep_alive = np.copy(self.arms_thresh)
        self.arm_pulled = np.zeros(self.num_arms, int)

    def choose_arm(self, user_context):
        """
        :input: the sampled user (integer in the range [0,num_users-1])
        :output: the chosen arm, content to show to the user (integer in the range [0,num_arms-1])
        """
        self.curr_round += 1
        i = self.curr_round
        self.user_selection[i] = user_context

        # print()
        # print("round: ", i)
        # print("Arms pulled: ", self.arm_pulled)
        # print("Threshold: ", self.keep_alive)
        # time.sleep(0.3)

        rounds_left = self.phase_len - np.sum(self.arm_pulled)
        if rounds_left == np.sum(self.keep_alive):
            # Emergency
            chosen_arm = self.keep_alive.argmax()
            # print()
            # print("----Emergency: ", chosen_arm)
        else:
            chosen_arm = self.dist_max[user_context].argmax()

        self.arm_selection[i] = chosen_arm

        if self.keep_alive[chosen_arm] > 0:
            self.keep_alive[chosen_arm] -= 1

        self.arm_pulled[chosen_arm] += 1
        #
        # print()
        # print("Rounds Left: ", rounds_left)
        # print("chosen:", chosen_arm)
        return chosen_arm

    def notify_outcome(self, reward):
        """
        :input: the sampled reward of the current round.
        """
        # TODO: Use this information for your algorithm
        i = self.curr_round
        user = int(self.user_selection[i])
        arm = int(self.arm_selection[i])

        self.content_ratings[user, arm] += reward
        self.dist_max[user, arm] = np.maximum(self.dist_max[user, arm], reward)

        self.norms.append(np.linalg.norm(self.dist_max))

        if not self.converged and (i % 2_000):
            # Check every 2000 rounds whether we have converged
            self.converged = self.check_convergence(self.norms, 200, 0.0001)

        # if i == 999_999:
        #     plt.hist(self.arm_selection)
        #     plt.show()
        # if i == self.num_rounds - 1:
        #     print()
        #     print(self.dist_max)
        #     print(self.content_ratings)

        # if i == 50000:
        #     print()
        #     print(np.sum(self.content_ratings, axis=0))
        #     raise ZeroDivisionError

        if (self.curr_round + 1) % self.phase_len == 0:
            self.reset_keep_alive()

    def check_convergence(self, array, window_size, threshold, plot=False):
        moving_avg = np.convolve(array, np.ones(window_size) / window_size, mode='valid')
        #plt.plot(array)
        if plot:
            plt.plot(moving_avg)
            plt.xlabel('Iteration')
            plt.ylabel('Moving Average')
            plt.show()

        diff = np.diff(moving_avg)
        return diff[-1] < threshold

    def get_id(self):
        # TODO: Make sure this function returns your ID, which is the name of this file!
        return "id_123456789_987654321"
