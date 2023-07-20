import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import chain, combinations
import time

EXPLORE = "explore"
EXPLOIT = "exploit"


def powerset(iterable):
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def find_best_arms(config, rounds):
    arms_comb = powerset(range(config["num_arms"]))
    best_arms = 0
    best_val = -np.inf
    begin = time.time()
    for arms in arms_comb:
        if not arms:
            continue
        arms = list(arms)
        num_arms = len(arms)

        ERM = config["ERM"][:, arms]
        thresh = config["arms_thresh"][arms]
        # print(arms)
        # print(ERM)
        # print(thresh)
        # print("---")

        params = [rounds, config["phase_len"],
                 num_arms, config["num_users"],
                 thresh, config["users_distribution"],
                 ERM]

        sim = FakeSimulation(*params)
        planner = Planner(rounds, config["phase_len"],
                 num_arms, config["num_users"],
                 thresh, config["users_distribution"], Testing=True)
        val = sim.simulation(planner)
        # print(f"\nArms: {arms} yielded: {val}")
        if val > best_val:
            best_val = val
            best_arms = arms
    print()
    print(f"Best arms are {best_arms} with value: {best_val}")
    print("\t\t took: ", time.time() - begin)
    return np.array(best_arms)


def argmax_include(arr, indices):
    if indices is None:
        return arr.argmax()
    subarray = arr[indices]
    return indices[np.argmax(subarray)]


class FakeSimulation:
    def __init__(self, num_rounds, phase_len, num_arms, num_users, arms_thresh, users_distribution, ERM):
        """
        :input: num_rounds - number of rounds
                phase_len - number of rounds at each phase
                num_arms - number of content providers
                num_users - number of users
                arms_thresh - the exposure demands of the content providers (np array of size num_arms)
                users_distribution - the probabilities of the users to arrive (np array of size num_users)
                ERM - expected reward matrix (2D np array of size (num_arms x num_users))
                      ERM[i][j] is the expected reward of user i from content provider j
        """
        self.num_rounds = num_rounds
        self.phase_len = phase_len
        self.num_arms = num_arms
        self.num_users = num_users
        self.arms_thresh = arms_thresh
        self.users_distribution = users_distribution
        self.ERM = ERM
        self.exposure_list = np.zeros(self.num_arms)
        self.inactive_arms = set()  # set of arms that left the system

    def sample_user(self):
        """
        :output: the sampled user, an integer in the range [0,self.num_users-1]
        """
        return int(np.random.choice(range(self.num_users), size=1, p=self.users_distribution))

    def sample_reward(self, sampled_user, chosen_arm):
        """
        :input: sampled_user - the sampled user
                chosen_arm - the content provider that was recommended to the user
        :output: the sampled reward
        """
        if chosen_arm >= self.num_arms or chosen_arm in self.inactive_arms:
            return 0
        else:
            return np.random.uniform(0, 2 * self.ERM[sampled_user][chosen_arm])

    def deactivate_arms(self):
        """
        this function is called at the end of each phase and deactivates arms that haven't gotten enough exposure
        (deactivated arm == arm that has departed)
        """
        for arm in range(self.num_arms):
            if self.exposure_list[arm] < self.arms_thresh[arm]:
                if arm not in self.inactive_arms:
                    pass
                self.inactive_arms.add(arm)
        self.exposure_list = np.zeros(self.num_arms)  # initiate the exposure list for the next phase.

    def simulation(self, planner, with_deactivation=True):
        """
        :input: the recommendation algorithm class implementation
        :output: the total reward for the algorithm
        """
        total_reward = 0
        for i in range(self.num_rounds):
            user_context = self.sample_user()
            chosen_arm = planner.choose_arm(user_context)
            reward = self.sample_reward(user_context, chosen_arm)
            planner.notify_outcome(reward)
            total_reward += reward
            self.exposure_list[chosen_arm] += 1

            if (i + 1) % self.phase_len == 0 and with_deactivation:  # satisfied only when it is the end of the phase
                self.deactivate_arms()

        return total_reward


class Planner:
    def __init__(self, num_rounds, phase_len, num_arms, num_users, arms_thresh, users_distribution, Testing=False):
        """
        :input: the instance parameters (see explanation in MABSimulation constructor)
        """
        self.num_rounds = num_rounds
        self.phase_len = phase_len
        self.num_arms = num_arms
        self.num_users = num_users
        self.arms_thresh = arms_thresh.astype(int)
        self.user_distribution = users_distribution
        self.Testing = Testing

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
        self.best_combo = None

    def reset_keep_alive(self):
        self.keep_alive = np.copy(self.arms_thresh)
        self.arm_pulled = np.zeros(self.num_arms, int)

    def get_config(self):
        return {
            'num_rounds': self.num_rounds,
            'phase_len': self.phase_len,
            'num_arms': self.num_arms,
            'num_users': self.num_users,
            'users_distribution': np.copy(self.user_distribution),
            'arms_thresh': np.copy(self.arms_thresh),
            'ERM': np.copy(self.dist_max / 2)
        }

    def update_config(self):
        print("--", np.array(range(self.num_arms)))
        print("--",self.best_combo)
        deactivate_arms = np.setdiff1d(np.array(range(self.num_arms)), self.best_combo)
        print("--------------", deactivate_arms)
        self.arms_thresh[deactivate_arms] = 0
        self.keep_alive[deactivate_arms] = 0

    def choose_arm(self, user_context):
        """
        :input: the sampled user (integer in the range [0,num_users-1])
        :output: the chosen arm, content to show to the user (integer in the range [0,num_arms-1])
        """
        self.curr_round += 1
        i = self.curr_round
        self.user_selection[i] = user_context

        rounds_left = self.phase_len - np.sum(self.arm_pulled)
        if rounds_left == np.sum(self.keep_alive):
            # Emergency
            chosen_arm = argmax_include(self.keep_alive, self.best_combo)
            # print()
            # print("----Emergency: ", chosen_arm)
        else:
            chosen_arm = argmax_include(self.dist_max[user_context], self.best_combo)

        if self.best_combo is not None:
            assert chosen_arm in self.best_combo

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
        i = self.curr_round
        user = int(self.user_selection[i])
        arm = int(self.arm_selection[i])

        self.content_ratings[user, arm] += reward

        self.dist_max[user, arm] = np.maximum(self.dist_max[user, arm], reward)

        self.norms.append(np.linalg.norm(self.dist_max))

        if i > 1 and not self.converged and (i % 200) == 0 and not self.Testing:
            window_size = 200
            moving_avg = np.convolve(self.norms,
                                     np.ones(window_size) / window_size,
                                     mode='valid')
            if abs(moving_avg[-2] - moving_avg[-1]) < 10 ** -5:
                self.converged = True
                self.best_combo = find_best_arms(self.get_config(), 20_000)
                self.update_config()

        if (self.curr_round + 1) % self.phase_len == 0:
            self.reset_keep_alive()


    def get_id(self):
        return "id_206528382_323958140"
