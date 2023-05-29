import numpy as np
from fake_sim import FakeSimulation
from id_123456789_987654321 import Planner


NUM_ROUNDS = 10 ** 6
PHASE_LEN = 10 ** 2
TIME_CAP = 2 * (10 ** 2)


np.set_printoptions(suppress=True)
phase = 100
p = 0.5
arms_thresh = np.array([0.1, 0.6])
users_distribution = np.array([p, 1-p])
reward_matrix = np.array([[0.5, 0],
                          [0, 0.5]
                          ])

expected_rewards = users_distribution[:, np.newaxis] * reward_matrix
expected_reward_per_arm = np.sum(expected_rewards, axis=0)
# print(np.multiply(expected_reward_per_arm, arms_thresh))

from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def find_best_arms(config, rounds):
    arms_comb = powerset(range(config["num_arms"]))
    best_arms = 0
    best_val = -np.inf
    for arms in arms_comb:
        if not arms:
            continue
        arms = list(arms)
        num_arms = len(arms)

        ERM = config["ERM"][:, arms]
        thresh = config["arms_thresh"][arms]
        print(arms)
        print(ERM)
        print(thresh)
        print("---")
        params = [rounds, config["phase_len"],
                 num_arms, config["num_users"],
                 thresh, config["users_distribution"],
                 ERM]

        sim = FakeSimulation(*params)
        planner = Planner(rounds, config["phase_len"],
                 num_arms, config["num_users"],
                 thresh, config["users_distribution"])
        val = sim.simulation(planner)
        print(f"Arms: {arms} yielded: {val}")
        if val > best_val:
            best_val = val
            best_arms = arms
    print()
    print(f"Best arms are {best_arms} with value: {best_val}")


config = {
    'num_rounds': NUM_ROUNDS,
    'phase_len': PHASE_LEN,
    'num_arms': 3,
    'num_users': 2,
    'users_distribution': np.array([0.6, 0.4]),
    'arms_thresh': np.array([0, 0.4, 0.4]) * PHASE_LEN,
    'ERM': np.array([[0.5, 0, 0],
                     [0, (1 + (4 * (NUM_ROUNDS ** (-1 / 3)))) / 2, 1 / 2]])
}
find_best_arms(config, 10_000)
