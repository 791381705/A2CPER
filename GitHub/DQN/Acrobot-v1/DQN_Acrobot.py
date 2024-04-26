import gym
import numpy as np
import matplotlib.pyplot as plt
import time


class DQNAgent:
    def __init__(self, env, alpha=0.001, gamma=0.99):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        self.alpha = alpha
        self.gamma = gamma

        self.weights = np.random.rand(self.obs_dim, self.act_dim)
        self.episode_rewards = []

    def q_values(self, state):
        return np.dot(state, self.weights)

    def update(self, state, action, reward, next_state):
        q_values_next = self.q_values(next_state)
        td_error = reward + self.gamma * np.max(q_values_next) - self.q_values(state)[action]

        # Update weights
        self.weights[:, action] += self.alpha * td_error * state


def plot_durations(episode_count, episode_durations):
    plt.figure(2)
    plt.clf()
    plt.plot(episode_count, np.array(episode_durations) * 10)  # Multiply y-values by 10
    plt.xlabel('Episode')
    plt.ylabel('Time (seconds)')
    plt.title('DQN_Acrobot Time')
    plt.pause(0.001)


def main():
    env = gym.make('Acrobot-v1')
    env.seed(0)
    agent = DQNAgent(env)

    num_episodes = 1000
    num_steps = 200

    episode_count = []
    episode_durations = []

    plt.ion()

    for episode in range(num_episodes):
        state = env.reset()
        start_time = time.time()
        for step in range(num_steps):
            action = np.argmax(agent.q_values(state))

            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state

            if done:
                end_time = time.time()
                episode_count.append(episode + 1)
                episode_durations.append(end_time - start_time)
                print(end_time - start_time)
                plot_durations(episode_count, episode_durations)

                break

    # Save plot as an image file
    path = './DQN_Acrobot_Time/DQN_Acrobot_Time.png'
    plt.savefig(path)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
