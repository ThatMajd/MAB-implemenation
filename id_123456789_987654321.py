import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

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
        self.curr_round = 0

        self.norms = []
        self.converged = False

        self.mode = EXPLORE
        
    def reset_keep_alive(self):
        self.keep_alive = np.copy(self.arms_thresh)
        self.arm_pulled = np.zeros(self.num_arms, int)

    def choose_arm(self, user_context):
        """
        :input: the sampled user (integer in the range [0,num_users-1])
        :output: the chosen arm, content to show to the user (integer in the range [0,num_arms-1])
        """
        i = self.curr_round
        self.user_selection[i] = user_context

        time.sleep(0.2)
        print()
        print("round: ", i+1)
        print("Arms pulled: ", self.arm_pulled)
        print("Threshold: ", self.keep_alive)
        print("Rounds Left: ", self.phase_len - (np.sum(self.arm_pulled)))
        if self.phase_len - (np.sum(self.arm_pulled)) == np.sum(self.keep_alive):
            # Emergency
            chosen_arm = self.keep_alive.argmax()
            print(self.keep_alive)
            print(chosen_arm)
        else:
            chosen_arm = self.dist_max[user_context].argmax()

        if (self.curr_round+1) % self.phase_len == 0:
            self.reset_keep_alive()
            self.mode = EXPLORE

        if i == 101:
            print()
            print(self.keep_alive)
            print(self.arm_pulled)

        self.arm_selection[i] = chosen_arm
        self.arm_pulled[chosen_arm] += 1
        if self.keep_alive[chosen_arm] > 0:
            self.keep_alive[chosen_arm] -= 1
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

        # self.dist_min[user, arm] = np.minimum(self.dist_min[user, arm], reward)
        self.dist_max[user, arm] = np.maximum(self.dist_max[user, arm], reward)

        self.norms.append(np.linalg.norm(self.dist_max))
        # TODO currently can't find a better way to test convergence
        if i == 2500:
            self.converged = True


        if (i+1) % 100000 == 0:
            print("-----rewards-----")
            print(self.dist_max)
        plot = False
        if (i+1) == 1000 and plot:
            steps = np.arange(len(self.norms)-1)
            plt.plot(steps, np.abs(np.diff(self.norms)))
            plt.xlabel('Time Steps')
            plt.ylabel('Norm Value')
            plt.title('Norm Values over Time')
            plt.grid(True)
            plt.show()
        self.curr_round += 1


    def get_id(self):
        # TODO: Make sure this function returns your ID, which is the name of this file!
        return "id_123456789_987654321"
