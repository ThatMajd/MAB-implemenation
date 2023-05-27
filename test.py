import numpy as np

a = np.array([[0.49999633, 0.19999772],
 [0.39998158, 0.69999138]])

print(a)
print(a + 0.0001)

b = (a + 0.0001) - a
print(np.linalg.norm(a))

class LinUCB:
    def __init__(self, num_actions, num_features, alpha=0.0001):
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


        p += (np.random.random(len(p)) * 0.000001)
        # Choose the action with the highest UCB
        #print(p)
        chosen_action = np.argmax(p)
        return chosen_action

    def update(self, action, context, reward):
        x = np.atleast_2d(context).T  # Reshape the context as a column vector
        self.A[int(action)] += np.dot(x, x.T)
        self.b[int(action)] += reward * x