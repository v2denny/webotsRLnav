import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
from environmentv5 import WebotsEnv

# Initialize the environment
env = WebotsEnv()

# Load the trained model
model_path = '900kv5.zip'  # Adjust this path if necessary
model = PPO.load(model_path, env=env)

# Parameters for testing
num_episodes = 10  # Number of episodes to test
max_steps_per_episode = 1000  # Maximum steps per episode

# Run the testing loop
for episode in range(num_episodes):
    obs, _ = env.reset()
    total_rewards = 0

    for step in range(max_steps_per_episode):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, truncated, _ = env.step(action)
        total_rewards += rewards

        if dones or truncated:
            break

    print(f"Episode {episode + 1}: Total Reward: {total_rewards}")

# Optionally, close the environment if it has a close method
if hasattr(env, 'close'):
    env.close()
