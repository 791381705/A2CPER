import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
from collections import deque
import random


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1)

    def forward(self, x):
        return self.multihead_attn(x, x, x)[0]


class ActorCriticWithAttention(nn.Module):
    def __init__(self, obs_dim, act_dim, n_heads=1):
        super(ActorCriticWithAttention, self).__init__()
        self.embedding = nn.Linear(obs_dim, 128)
        self.attention = SelfAttention(embed_dim=128, num_heads=n_heads)

        self.actor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        state_emb = self.embedding(state).unsqueeze(1)  # Add sequence dimension
        attn_output = self.attention(state_emb)
        attn_output = attn_output.squeeze(1)  # Remove sequence dimension

        action_probs = self.actor(attn_output)
        state_value = self.critic(attn_output)
        return action_probs, state_value


class PrioritizedMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.epsilon = 1e-6

    def add(self, error, sample):
        priority = (abs(error) + self.epsilon) ** 0.6
        self.buffer.append(sample)
        self.priorities.append(priority)

    def sample(self, batch_size):
        probabilities = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(range(len(self.buffer)), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        return map(np.array, zip(*samples))

    def __len__(self):
        return len(self.buffer)


def compute_td_error(model, target_model, state, action, reward, next_state, done, gamma):
    state = torch.FloatTensor(state).unsqueeze(0)
    next_state = torch.FloatTensor(next_state).unsqueeze(0)
    action = torch.tensor([[action]], dtype=torch.int64)
    reward = torch.FloatTensor([reward])
    done = torch.FloatTensor([not done])

    _, current_val = model(state)

    _, next_val = target_model(next_state)

    target = reward + gamma * next_val * done

    td_error = target - current_val
    return td_error.detach().abs()


def update(model, target_model, optimizer, memory, gamma, tau, batch_size):
    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).view(-1, 1)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    _, next_state_values = target_model(next_states)
    next_state_values = next_state_values.squeeze(-1)

    target_values = rewards + gamma * next_state_values * (1 - dones)

    action_probs, state_values = model(states)
    state_values = state_values.squeeze(-1)

    dist = torch.distributions.Categorical(action_probs)
    log_probs = dist.log_prob(actions.squeeze(-1))

    advantages = target_values.detach() - state_values
    actor_loss = -(log_probs * advantages).mean()
    critic_loss = advantages.pow(2).mean()

    optimizer.zero_grad()
    (actor_loss + critic_loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    for target_param, local_param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def train(env, model, target_model, num_episodes, batch_size, gamma=0.99, tau=0.005):
    optimizer = optim.Adam(model.parameters())
    memory = PrioritizedMemory(capacity=10000)
    episode_rewards = []

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'r-', linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    plt.title("Episode Rewards")

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs, _ = model(state_tensor)
            action = torch.distributions.Categorical(action_probs).sample().item()

            next_state, reward, done, _ = env.step(action)
            error = compute_td_error(model, target_model, state, action, reward, next_state, done, gamma)
            memory.add(error, (state, action, reward, next_state, done))

            state = next_state
            episode_reward += reward

            if len(memory) >= batch_size:
                update(model, target_model, optimizer, memory, gamma, tau, batch_size)

            if done:
                episode_rewards.append(episode_reward)
                line.set_xdata(np.append(line.get_xdata(), episode))
                line.set_ydata(np.append(line.get_ydata(), episode_reward))
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
                print(f"Episode {episode + 1}, Total Reward: {episode_reward}")
                break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    env = gym.make('Acrobot-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = ActorCriticWithAttention(obs_dim, act_dim, n_heads=1)
    target_model = copy.deepcopy(model)
    train(env, model, target_model, num_episodes=500, batch_size=64, gamma=0.99, tau=0.005)
