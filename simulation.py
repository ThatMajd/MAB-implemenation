import numpy as np
from tqdm import tqdm
import time
from id_123456789_987654321 import Planner

NUM_ROUNDS = 10 ** 6
PHASE_LEN = 10 ** 2
TIME_CAP = 2 * (10 ** 2)


class MABSimulation:
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
        # exposure_list[i] represents the number of exposures arm i has gotten in the current phase
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
                print("------------------")
                print(self.exposure_list[arm])
                print(self.arms_thresh[arm])
                if arm not in self.inactive_arms: print("\n arm " + str(arm) + " is deactivated!")
                self.inactive_arms.add(arm)
        self.exposure_list = np.zeros(self.num_arms)  # initiate the exposure list for the next phase.

    def simulation(self, planner, with_deactivation=True):
        """
        :input: the recommendation algorithm class implementation
        :output: the total reward for the algorithm
        """
        total_reward = 0
        begin_time = time.time()
        for i in tqdm(range(self.num_rounds)):
            user_context = self.sample_user()
            chosen_arm = planner.choose_arm(user_context)
            reward = self.sample_reward(user_context, chosen_arm)
            planner.notify_outcome(reward)
            total_reward += reward
            self.exposure_list[chosen_arm] += 1

            if (i + 1) % self.phase_len == 0 and with_deactivation:  # satisfied only when it is the end of the phase
                self.deactivate_arms()

        print("run took: ", time.time() - begin_time, TIME_CAP)
        if time.time() - begin_time > TIME_CAP:
            print("the planner operation is too slow")
            return 0

        return total_reward


def get_simulation_params(simulation_num):
    """
    :input: the ID of the simulation you want to run
    :output: the simulation parameters
    """
    simulations = [
        {
            'num_rounds': NUM_ROUNDS,
            'phase_len': PHASE_LEN,
            'num_arms': 2,
            'num_users': 2,
            'users_distribution': np.array([0.5, 0.5]),
            'arms_thresh': np.array([0.1, 0.6]) * PHASE_LEN,
            'ERM': np.array([[0.5, 0], [0, 0.5]]),
        },
        {
            'num_rounds': NUM_ROUNDS,
            'phase_len': PHASE_LEN,
            'num_arms': 2,
            'num_users': 2,
            'users_distribution': np.array([0.1, 0.9]),
            'arms_thresh': np.array([0.6, 0.1]) * PHASE_LEN,
            'ERM': np.array([[0.5, 0], [0, 0.5]])
        },
        {
            'num_rounds': NUM_ROUNDS,
            'phase_len': PHASE_LEN,
            'num_arms': 2,
            'num_users': 2,
            'users_distribution': np.array([0.5, 0.5]),
            'arms_thresh': np.array([0.48, 0.48]) * PHASE_LEN,
            'ERM': np.array([[0.5, 0], [0, 0.5]])
        },
        {
            'num_rounds': NUM_ROUNDS,
            'phase_len': PHASE_LEN,
            'num_arms': 3,
            'num_users': 3,
            'users_distribution': np.array([0.5, 0.25, 0.25]),
            'arms_thresh': np.array([0, 0.33, 0.33]) * PHASE_LEN,
            'ERM': np.array([[1, 0.5, 0], [0, 2 / 3, 1 / 2], [0, 0, 1 / 2]])
        },
        {
            'num_rounds': NUM_ROUNDS,
            'phase_len': PHASE_LEN,
            'num_arms': 3,
            'num_users': 2,
            'users_distribution': np.array([0.6, 0.4]),
            'arms_thresh': np.array([0, 0.4, 0.4]) * PHASE_LEN,
            'ERM': np.array([[0.5, 0, 0], [0, (1 + (4 * (NUM_ROUNDS ** (-1 / 3)))) / 2, 1 / 2]])
        },
    ]
    return simulations[simulation_num]


def run_simulation(simulation_num):
    """
    :input: simulation ID
    :output: the reward of the students' planners for the given simulation
    """
    params = get_simulation_params(simulation_num)

    mab = MABSimulation(**params)

    planner = Planner(params['num_rounds'], params['phase_len'], params['num_arms'], params['num_users'],
                      params['arms_thresh'], params['users_distribution'])

    print('planner ' + planner.get_id() + ' is currently running')
    reward = mab.simulation(planner)

    return reward


def main():
    reward = run_simulation(1)
    print("The total reward of your planner is " + str(reward))


if __name__ == '__main__':
    main()
