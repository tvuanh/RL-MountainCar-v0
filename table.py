from collections import deque

import numpy as np

import gym


class Controller(object):

    def __init__(self, lows, highs, n_bins, n_actions, gamma=1.0, epsilon=0.02, lamb=0.5):
        self.lows = lows
        self.highs = highs
        self.n_bins = n_bins
        self.n_actions = n_actions
        self.Qtable = np.zeros((n_bins, n_bins, n_actions))
        self.trace = np.zeros((n_bins, n_bins, n_actions))
        self.gamma = gamma
        self.epsilon = epsilon
        self.lamb = lamb

    def discretize(self, state):
        steps = (self.highs - self.lows) / self.n_bins
        return np.array((state - self.lows) / steps, dtype=np.int)

    def optimal_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            ix, iy = self.discretize(state)
            logits = np.exp(self.Qtable[ix, iy])
            logits /= np.sum(logits)
            return np.random.choice(self.n_actions, p=logits)

    def train(self, state, action, reward, next_state, alpha):
        ix, iy = self.discretize(state)
        self.trace[ix, iy, action] = 1
        next_ix, next_iy = self.discretize(next_state)
        delta = reward + self.gamma * np.max(self.Qtable[next_ix, next_iy]) - self.Qtable[ix, iy, action]
        self.Qtable += alpha * delta * self.trace
        self.trace *= self.gamma * self.lamb


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    env.seed(0)
    # np.random.seed(0)

    controller = Controller(
        lows=env.observation_space.low,
        highs=env.observation_space.high,
        n_bins=20, n_actions=env.action_space.n)

    scores = deque(maxlen=100)
    benchmark = -110.
    episodes = 20000
    initial_alpha = 1.0 # Learning rate
    min_alpha = 0.01

    for episode in range(episodes):
        state = env.reset()
        rewards = 0
        alpha = max(min_alpha, initial_alpha * (0.85 ** (episode // 100)))
        done = False
        while not done:
            action = controller.optimal_action(state)
            a, b = controller.discretize(state)
            next_state, reward, done, _ = env.step(action)
            rewards += reward
            controller.train(state, action, reward, next_state, alpha)
            state = next_state

        scores.append(rewards)

        print("episode {} rewards {} average score {}".format(episode, rewards, np.mean(scores)))

        if np.mean(scores) >= benchmark:
            break
