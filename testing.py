from environment import WebotsEnv as DiscreteEnv  # For PPO and DQN
from environment_cont import WebotsEnv as ContinuousEnv  # For SAC and TD3
from stable_baselines3 import PPO, DQN, SAC, TD3
import os
import time
from datetime import datetime


def test_agent(algorithm="PPO", mode="start", model_path=None, num_episodes=20, save_folder="benchmarking"):
    """
    Test a trained agent and save metrics

    Args:
        algorithm: Algorithm name (PPO, DQN, SAC, TD3)
        mode: Test mode (start, easy, medium, hard, etc.)
        model_path: Path to the trained model
        num_episodes: Number of episodes to test
        save_folder: Folder to save results (default: benchmarking)
    """

    if not model_path or not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' does not exist!")
        return None, None, None

    # Create appropriate environment based on algorithm
    if algorithm.upper() in ['PPO', 'DQN']:
        env = DiscreteEnv()
    elif algorithm.upper() in ['SAC', 'TD3']:
        env = ContinuousEnv()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    env.trainmode = mode

    # Create save directories
    os.makedirs(save_folder, exist_ok=True)

    # Load the model
    try:
        if algorithm.upper() == 'PPO':
            model = PPO.load(model_path, env=env)
        elif algorithm.upper() == 'DQN':
            model = DQN.load(model_path, env=env)
        elif algorithm.upper() == 'SAC':
            model = SAC.load(model_path, env=env)
        elif algorithm.upper() == 'TD3':
            model = TD3.load(model_path, env=env)

        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None, None

    # Reset environment metrics for testing - use the correct MetricsTracker class
    if algorithm.upper() in ['PPO', 'DQN']:
        # Discrete environment - import MetricsTracker from environment.py
        from environment import MetricsTracker as DiscreteMetricsTracker
        env.metrics = DiscreteMetricsTracker()
    else:
        # Continuous environment - import MetricsTracker from environment_cont.py
        from environment_cont import MetricsTracker as ContinuousMetricsTracker
        env.metrics = ContinuousMetricsTracker()

    print(f"\n######## TESTING {algorithm.upper()} ########")
    print(f"Mode: {mode}")
    print(f"Episodes: {num_episodes}")
    print(f"Model: {model_path}")
    print("=" * 50)

    # Run test episodes
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        truncated = False

        print(f"Episode {episode + 1}/{num_episodes}: ", end="")

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)

            # Convert numpy array action to integer for discrete environments
            if algorithm.upper() in ['PPO', 'DQN']:
                # For discrete action spaces, convert numpy array to int
                if hasattr(action, 'item'):
                    action = action.item()  # Convert numpy array to scalar
                elif hasattr(action, '__len__') and len(action) == 1:
                    action = int(action[0])  # Convert single-element array to int
                else:
                    action = int(action)  # Fallback conversion

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        # Get outcome from environment metrics
        if len(env.metrics.episode_successes) > 0:
            outcome = env.metrics.episode_successes[-1]
        else:
            outcome = 'Unknown'

        print(f"Reward={total_reward:.1f}, Steps={steps}, Outcome={outcome}")

    # Create filename for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_filename = f"{algorithm.lower()}_{mode}_{num_episodes}ep_{timestamp}"
    test_filepath = os.path.join(save_folder, f"{test_filename}.csv")

    # Save metrics using environment's built-in method
    summary = env.save_metrics(test_filepath)

    # Print test results
    print("\n######## TEST RESULTS ########")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    print(f"\nResults saved to: {test_filepath}")

    return env, model, summary


# Main execution
if __name__ == "__main__":
    # Test configuration - MODIFY THESE VALUES
    test_config = {
        'algorithm': 'TD3',  # PPO, DQN, SAC, TD3
        'mode': 'random',  # start, easy, medium, hard, all, random
        'model_path': 'models/td3_all_600000.zip',  # Path to your trained model
        'num_episodes': 100,  # Number of test episodes
        'save_folder': 'benchmarking/level_random'  # Folder to save results
    }

    # Run the test
    env, model, summary = test_agent(
        algorithm=test_config['algorithm'],
        mode=test_config['mode'],
        model_path=test_config['model_path'],
        num_episodes=test_config['num_episodes'],
        save_folder=test_config['save_folder']
    )

    if summary:
        print(f"\nTesting completed successfully!")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Average Reward: {summary['avg_episode_reward']:.2f}")
        print(f"Average Steps: {summary['avg_episode_steps']:.1f}")
    else:
        print("Testing failed!")