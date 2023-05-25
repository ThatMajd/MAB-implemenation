import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from test import LinUCB

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
        self.arms_thresh = arms_thresh
        self.user_distribution = users_distribution
        self.num_features = 4
        self.model = LinUCB(self.num_arms, self.num_features)

        self.curr_round = 0
        self.actions = np.zeros(self.num_rounds)
        self.contexts = np.zeros(self.num_rounds)

    # def temp(self):
    #     eps = 0.2
    #     choices = np.zeros(n, dtype=int)
    #     rewards = np.zeros(n)
    #     explore = np.zeros(n)
    #     norms = np.zeros(n)
    #     b = np.zeros_like(th)
    #     A = np.zeros((n_a, k, k))
    #     for a in range(0, n_a):
    #         A[a] = np.identity(k)
    #     th_hat = np.zeros_like(th)  # our temporary feature vectors, our best current guesses
    #     p = np.zeros(n_a)
    #     alph = 0.2
    #
    #     # LINUCB, usign disjoint model
    #     # This is all from Algorithm 1, p 664, "A contextual bandit appraoch..." Li, Langford
    #     for i in range(0, n):
    #         x_i = D[i]  # the current context vector
    #         for a in range(0, n_a):
    #             A_inv = np.linalg.inv(A[a])  # we use it twice so cache it.
    #             th_hat[a] = A_inv.dot(b[a])  # Line 5
    #             ta = x_i.dot(A_inv).dot(x_i)  # how informative is this?
    #             a_upper_ci = alph * np.sqrt(ta)  # upper part of variance interval
    #
    #             a_mean = th_hat[a].dot(x_i)  # current estimate of mean
    #             p[a] = a_mean + a_upper_ci
    #         norms[i] = np.linalg.norm(th_hat - th, 'fro')  # diagnostic, are we converging?
    #         # Let's hnot be biased with tiebraks, but add in some random noise
    #         p = p + (np.random.random(len(p)) * 0.000001)
    #         choices[i] = p.argmax()  # choose the highest, line 11
    #
    #         # See what kind of result we get
    #         rewards[i] = th[choices[i]].dot(x_i)  # using actual theta to figure out reward
    #
    #         # update the input vector
    #         A[choices[i]] += np.outer(x_i, x_i)
    #         b[choices[i]] += rewards[i] * x_
    def choose_arm(self, user_context):
        """
        :input: the sampled user (integer in the range [0,num_users-1])
        :output: the chosen arm, content to show to the user (integer in the range [0,num_arms-1])
        """
        contexts = [np.random.rand(self.num_features) for _ in range(10)]
        contexts[0][self.num_features-1] = user_context
        return self.model.choose_action(contexts[0])

    def notify_outcome(self, reward):
        """
        :input: the sampled reward of the current round.
        """
        # TODO: Use this information for your algorithm
        self.model.update(self.actions[self.curr_round], self.contexts[self.curr_round], reward)
        self.curr_round += 1

    def get_id(self):
        # TODO: Make sure this function returns your ID, which is the name of this file!
        return "id_123456789_987654321"
