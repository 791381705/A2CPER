import gym
import numpy as np
import matplotlib.pyplot as plt
import time


class PolicyAgent:
    def __init__(self, env, alpha=0.001, gamma=0.99):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        self.alpha = alpha
        self.gamma = gamma

        self.weights = np.random.rand(self.obs_dim, self.act_dim)
        self.episode_rewards = []

    def policy(self, state):
        probabilities = np.exp(np.dot(state, self.weights))
        action = np.random.choice(self.act_dim, p=probabilities / np.sum(probabilities))
        return action

    def update(self, state, action, reward, next_state):
        td_target = reward + self.gamma * np.max(np.dot(next_state, self.weights))
        td_error = td_target - np.dot(state, self.weights[:, action])

        # Update weights
        self.weights[:, action] += self.alpha * td_error * state


def plot_durations(episode_count, episode_durations):
    plt.figure(2)
    plt.clf()
    plt.plot(episode_count, episode_durations )
    plt.xlabel('Episode')
    plt.ylabel('Time (seconds)')
    plt.title('Policy_Acrobot Time')
    plt.pause(0.001)


def main():
    env = gym.make('Acrobot-v1')
    env.seed(0)
    agent = PolicyAgent(env)

    num_episodes = 1000
    num_steps = 200

    episode_count = []
    episode_durations = []

    plt.ion()

    for episode in range(num_episodes):
        state = env.reset()
        start_time = time.time()
        episode_reward = 0
        for step in range(num_steps):
            action = agent.policy(state)

            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state

            episode_reward += reward

            if done:
                end_time = time.time()
                episode_count.append(episode + 1)
                episode_durations.append((end_time - start_time)*10)
                agent.episode_rewards.append(episode_reward)

                plot_durations(episode_count, episode_durations)

                break

    plt.ioff()

    path = './Polocy_Acrobot_Time/Polocy_Acrobot_Time.png'
    plt.savefig(path)
    plt.show()

if __name__ == "__main__":
    main()
