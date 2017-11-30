from collections import deque

import numpy as np

import gym


class Controller(object):

    def __init__(self, lows, highs, n_bins, n_actions, gamma=1.0, epsilon=0.02):
        self.lows = lows
        self.highs = highs
        self.n_bins = n_bins
        self.action_space = range(n_actions)
        self.Qmean = np.zeros((n_bins, n_bins, n_actions))
        self.gamma = gamma
        self.epsilon = epsilon

    def discretize(self, state):
        steps = (self.highs - self.lows) / self.n_bins
        return np.array((state - self.lows) / steps, dtype=np.int)

    def optimal_action(self, state):
        if np.random.uniform(0., 1.) < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            ix, iy = self.discretize(state)
            logits = np.exp(self.Qmean[ix, iy])
            logits = logits / np.sum(logits)
            return np.random.choice(self.action_space, p=logits)

    def train(self, state, action, reward, next_state, alpha):
        ix, iy = self.discretize(state)
        next_ix, next_iy = self.discretize(next_state)
        update = reward + self.gamma * np.max(self.Qmean[next_ix, next_iy, action])
        self.Qmean[ix, iy, action] += alpha * (update - self.Qmean[ix, iy, action])


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    env.seed(0)
    np.random.seed(0)

    controller = Controller(
        lows=env.observation_space.low,
        highs=env.observation_space.high,
        n_bins=40, n_actions=env.action_space.n)

    scores = deque(maxlen=100)
    episodes = 20000
    initial_alpha = 1.0 # Learning rate
    min_alpha = 0.003

    for episode in range(episodes):
        state = env.reset()
        rewards = 0
        alpha = max(min_alpha, initial_alpha * (0.85 ** (episode // 100)))
        done = False
        while not done:
            action = controller.optimal_action(state)
            next_state, reward, done, _ = env.step(action)
            rewards += reward
            state = next_state
            controller.train(state, action, reward, next_state, alpha)

        scores.append(rewards)

        if episode % 100 == 0:
            print("episode {} rewards {} average score {}".format(episode + 1, rewards, np.mean(scores)))
