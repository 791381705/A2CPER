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
            nn.Softmax(dim=-1),
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

def train(env, model, num_episodes, gamma=0.99):
    optimizer = optim.Adam(model.parameters())
    episode_rewards = []

    # Set up live plot
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    line, = ax.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards')

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            state = torch.FloatTensor(state).unsqueeze(0)
            probs, value = model(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, done, _ = env.step(action.item())
            episode_reward += reward

            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            _, next_value = model(next_state)

            returns = reward + gamma * next_value * (1 - int(done))

            advantage = returns - value

            actor_loss = -(dist.log_prob(action) * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()

            loss = actor_loss + critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state.squeeze(0).numpy()

            if done:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}, Total Reward: {episode_reward}")

        # Update live plot
        line.set_ydata(episode_rewards)
        line.set_xdata(range(episode + 1))
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ioff()  # Turn off interactive mode
    plt.show()

if __name__ == "__main__":
    env = gym.make('Acrobot-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = ActorCritic(obs_dim, act_dim)
    train(env, model, num_episodes=1000)  # Reduced number of episodes for quicker demonstration
