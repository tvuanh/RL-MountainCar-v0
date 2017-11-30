import random
from collections import deque

import numpy as np

import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def preprocess_state(state):
    return np.array([state])


class Controller(object):

    def __init__(self, n_input, n_output, n_hidden=30, gamma=1.0, batch_size=10, model_instances=2):
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.action_space = range(n_output)
        self.gamma = gamma
        self.batch_size = batch_size
        self.model_instances = model_instances
        self.memory = dict()

        # action neural network
        self.action_models = []
        for _ in range(self.model_instances):
            model = Sequential()
            model.add(
                Dense(
                    self.n_hidden, input_dim=self.n_input, activation="tanh",
                    kernel_initializer="zeros", use_bias=True
                    )
                )
            model.add(
                Dense(
                    self.n_output, activation="linear",
                    kernel_initializer="zeros", use_bias=True
                    )
                )
            model.compile(
                loss="mse", optimizer=Adam(lr=0.01, decay=0.01)
                )
            self.action_models.append(model)

    def memorize(self, state, action, reward, next_state, done):
        mem = self.memory.get(
            reward, deque(maxlen=10 * self.batch_size * self.model_instances)
        )
        mem.append(
        (
            preprocess_state(state),
            action, reward,
            preprocess_state(next_state),
            done
            )
        )
        self.memory[reward] = mem

    def random_action(self, state):
        return np.random.choice(self.action_space)

    def optimal_action(self, state):
        s = preprocess_state(state)
        Qs = [model.predict(s)[0] for model in self.action_models]
        # estimate means and sigmas then draw random events to smoothen the sampling
        means = np.mean(Qs, axis=0)
        sigmas = np.sqrt(np.var(Qs, axis=0, ddof=1) / self.model_instances)
        draws = means + sigmas * np.random.randn(self.n_output)
        return np.argmax(draws)

    def replay(self):
        for model in self.action_models:
            minibatch = self.prepare_minibatch()
            x_batch, y_batch = list(), list()
            for state, action, reward, next_state, done in minibatch:
                y_target = model.predict(state)
                y_target[0, action] = reward + (1 - done) * self.gamma * np.max(
                    model.predict(next_state)
                    )
                x_batch.append(state[0])
                y_batch.append(y_target[0])
            model.fit(
                np.array(x_batch), np.array(y_batch), batch_size=len(x_batch),
                verbose=False)

    def prepare_minibatch(self):
        minibatch = []
        for c in self.memory.keys():
            mem = self.memory[c]
            if len(mem) <= self.model_instances * self.batch_size:
                size = max(1, len(mem) / self.model_instances)
            else:
                size = self.batch_size
            minibatch += random.sample(mem, size)
        return minibatch


def play(episodes, verbose=False):
    env = gym.make("MountainCar-v0")
    controller = Controller(
        n_input=env.observation_space.shape[0], n_output=env.action_space.n)

    benchmark = -110.
    scores = deque(maxlen=100)

    episode = 0
    while episode < episodes:
        episode += 1
        state = env.reset()
        done = False
        intra_episode_total_reward = 0
        steps = 0
        while not done:
            if episode < 200:
                action = controller.random_action(state)
            else:
                action = controller.optimal_action(state)
            next_state, reward, done, _ = env.step(action)
            controller.memorize(state, action, reward, next_state, done)
            state = next_state
            intra_episode_total_reward += reward
            steps += 1
        scores.append(intra_episode_total_reward)

        if verbose:
            print(
                "episode {} steps {} score {} average score {}".format(
                    episode, steps, intra_episode_total_reward, np.mean(scores)
                    )
                )

        controller.replay()

        if np.mean(scores) >= benchmark:
            break

    return episode


if __name__ == "__main__":
    episodes = 5000
    nplays = 1
    results = np.array([play(episodes, verbose=True) for _ in range(nplays)])
    success = results < episodes
    print("Total number of successful plays is {}/{}".format(np.sum(success), nplays))
    print("Average number of episodes before success per play {}".format(np.mean(results[success])))
