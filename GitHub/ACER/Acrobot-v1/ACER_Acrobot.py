import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Softmax(),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

class Memory:
    def __init__(self):
        self.clear()

    def store(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def sample(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.next_states),
            np.array(self.dones)
        )

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

def update(model, optimizer, memory, gamma):
    states, actions, rewards, next_states, dones = memory.sample()

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).view(-1, 1)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    # Calculate next state values
    _, next_state_values = model(next_states)
    next_state_values = next_state_values.squeeze(-1)

    # Expected values are only updated for non-final states
    target_values = rewards + gamma * next_state_values * (1 - dones)

    # Calculate values and advantage
    _, state_values = model(states)
    state_values = state_values.squeeze(-1)
    advantages = target_values.detach() - state_values

    # Calculate log probabilities
    action_probs, _ = model(states)
    dist = torch.distributions.Categorical(action_probs)
    log_probs = dist.log_prob(actions.squeeze(-1))

    # Calculate losses
    actor_loss = -(log_probs * advantages).mean()
    critic_loss = advantages.pow(2).mean()

    # Backpropagation
    optimizer.zero_grad()
    total_loss = actor_loss + critic_loss
    total_loss.backward()
    optimizer.step()

def train(env, model, num_episodes, batch_size, gamma=0.99):
    optimizer = optim.Adam(model.parameters())
    episode_rewards = []
    memory = Memory()

    # Setup live plotting
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    line, = ax.plot(episode_rewards)
    plt.title("Episode Rewards")

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs = model(state_tensor)
            action = torch.distributions.Categorical(action_probs).sample().item()

            next_state, reward, done = env.step(action)
            memory.store(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if len(memory.states) >= batch_size:
                update(model, optimizer, memory, gamma)
                memory.clear()

            if done:
                episode_rewards.append(episode_reward)
                line.set_ydata(episode_rewards)
                line.set_xdata(range(len(episode_rewards)))
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

    model = ActorCritic(obs_dim, act_dim)
    train(env, model, num_episodes=1000, batch_size=24, gamma=0.99)
