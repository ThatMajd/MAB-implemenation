import numpy as np

class LinUCB:
    def __init__(self, num_actions, num_features, alpha=1.0):
        self.num_actions = num_actions
        self.num_features = num_features
        self.alpha = alpha

        # Initialize action-specific parameters
        self.A = [np.identity(num_features) for _ in range(num_actions)]
        self.b = [np.zeros((num_features, 1)) for _ in range(num_actions)]

    def choose_action(self, context):
        p = np.zeros(self.num_actions)

        # Calculate the Upper Confidence Bound (UCB) for each action
        for a in range(self.num_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = np.dot(A_inv, self.b[a])
            x = np.atleast_2d(context).T  # Reshape the context as a column vector
            p[a] = np.dot(theta.T, x) + self.alpha * np.sqrt(np.dot(np.dot(x.T, A_inv), x))

        # Choose the action with the highest UCB
        chosen_action = np.argmax(p)
        return chosen_action

    def update(self, action, context, reward):
        x = np.atleast_2d(context).T  # Reshape the context as a column vector
        self.A[int(action)] += np.dot(x, x.T)
        self.b[int(action)] += reward * x