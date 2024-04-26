import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy

# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100

env = gym.make("CartPole-v1")
env = env.unwrapped
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape


class Net(nn.Module):
    """docstring for Net"""

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(50, 30)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, NUM_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

class DQN():
    """docstring for DQN"""

    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # get a 1D array
        if np.random.randn() <= EPISILO:  # greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:  # random policy
            action = np.random.randint(0, NUM_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):

        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES + 1:NUM_STATES + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -NUM_STATES:])

        # q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def reward_func(env, x, x_dot, theta, theta_dot):
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward


def plot_durations(episode_durations, average_length,average_reward):
    plt.ion()
    plt.figure(2)
    plt.clf()
    plt.ylim(0, 500)
    duration_t = torch.FloatTensor(episode_durations)
    duration_age = torch.FloatTensor(average_length)
    duration_average = torch.FloatTensor(average_reward)

    plt.title('CartPole-v1 Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Time Step')
    # plt.plot(duration_t.numpy())
    # plt.plot(duration_average.numpy(),color='r')
    plt.plot(duration_age.numpy(),color='r')


    plt.pause(0.00001)
    RunTime = len(episode_durations)
    path = './DQN_CartPole-v1_reward/' + 'DQNRunTime' + str(RunTime) + '.jpg'
    if len(episode_durations) % 200 == 0:
        plt.savefig(path)
    plt.pause(0.0000001)

def main():
    dqn = DQN()
    episodes = 2000
    print("Collecting Experience....")
    running_reward = 10
    live_time = []
    episode_rewards = []

    ac_age = []
    for i in range(episodes):
        state = env.reset()
        ep_reward = 0
        while True:
            # env.render()
            action = dqn.choose_action(state)
            next_state, _, done, info = env.step(action)
            x, x_dot, theta, theta_dot = next_state
            reward = reward_func(env, x, x_dot, theta, theta_dot)

            dqn.store_transition(state, action, reward, next_state)
            ep_reward += reward

            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()
                if done :
                    if ep_reward <= 500:
                        running_reward = running_reward * 0.99 + ep_reward * 0.01
                        live_time.append(ep_reward)

                        ac_age.append(running_reward)
                        # average_reward = np.mean(live_time)
                        episode_rewards.append(1)

                        plot_durations(live_time, ac_age,episode_rewards)
                        print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
                    else:
                        running_reward = running_reward * 0.99 + 500 * 0.01
                        live_time.append(500)
                        ac_age.append(running_reward)
                        # average_reward = np.mean(live_time)
                        episode_rewards.append(1)
                        plot_durations(live_time, ac_age, episode_rewards)
            else:
                if done :
                    running_reward = running_reward * 0.99 + ep_reward * 0.01
                    live_time.append(ep_reward)
                    ac_age.append(running_reward)
                    # average_reward = np.mean(live_time)
                    episode_rewards.append(1)
                    plot_durations(live_time, ac_age, episode_rewards)
            if done :
                break
            state = next_state
        r = copy.copy(reward)


if __name__ == '__main__':
    main()
