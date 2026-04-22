import numpy as np
import random
import pickle

class QLearningAgent:
    def __init__(self, state_shape=(3,3,3,3,3,3), num_actions=5,
                 alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.num_actions   = num_actions
        self.q_table       = np.zeros(state_shape + (num_actions,))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        current_q  = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        self.q_table[state][action] = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path="model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Model saved to {path}")

    def load(self, path="model.pkl"):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)
        self.epsilon = 0.0   # no exploration after loading
        print(f"Model loaded from {path}")