from environment import WebotsEnv as DiscreteEnv  # For PPO and DQN
from environment_cont import WebotsEnv as ContinuousEnv  # For SAC and TD3
import random
from stable_baselines3 import PPO, DQN
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
import os
import time
from datetime import datetime


# Custom callback to track metrics but not save intermediate files
class MetricsCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(MetricsCallback, self).__init__(verbose)

    def _on_step(self):
        # No intermediate saving, just track progress
        if self.n_calls % 10000 == 0 and self.verbose > 0:
            print(f"Training progress: {self.n_calls} steps completed")
        return True


# Main training code
def train_agent(algorithm="PPO", mode="start", total_steps=50000, model_path=None, metrics_prefix=None):
    # Create appropriate environment based on algorithm
    if algorithm.upper() in ['PPO', 'DQN']:
        env = DiscreteEnv()
        print(f"Using discrete environment for {algorithm}")
    elif algorithm.upper() in ['SAC', 'TD3']:
        env = ContinuousEnv()
        print(f"Using continuous environment for {algorithm}")
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    env.trainmode = mode

    # Track total steps including previous training if model is loaded
    cumulative_steps = total_steps

    # Create directories - simplified and consistent paths
    model_dir = "./models/"
    log_dir = "./logs/"
    metrics_dir = "./metrics/"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Create experiment name first for consistent naming
    experiment_name = f"{algorithm.lower()}_{mode}_{cumulative_steps}"

    # Create specific log directory for this run to avoid _1, _2 suffixes
    specific_log_dir = os.path.join(log_dir, experiment_name)
    if os.path.exists(specific_log_dir):
        # Remove previous log directory with same name if it exists
        import shutil
        shutil.rmtree(specific_log_dir)

    # Create or load model based on algorithm
    if model_path and os.path.exists(model_path):
        # Load existing model based on algorithm
        try:
            if algorithm.upper() == "PPO":
                model = PPO.load(model_path, env=env, tensorboard_log=log_dir)
            elif algorithm.upper() == "DQN":
                model = DQN.load(model_path, env=env, tensorboard_log=log_dir)
            elif algorithm.upper() == "SAC":
                model = SAC.load(model_path, env=env, tensorboard_log=log_dir)
            elif algorithm.upper() == "TD3":
                model = TD3.load(model_path, env=env, tensorboard_log=log_dir)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            print(f"Model loaded from {model_path}")

            # Extract previous steps from filename if possible
            try:
                # Attempt to extract steps from filename format "algo_mode_steps.zip"
                filename_parts = os.path.basename(model_path).split('_')
                if len(filename_parts) >= 3:
                    prev_steps = int(filename_parts[2].split('.')[0])
                    cumulative_steps = prev_steps + total_steps
                    # Update experiment name with new cumulative steps
                    experiment_name = f"{algorithm.lower()}_{mode}_{cumulative_steps}"
                    print(f"Continuing training from step {prev_steps}, will train to {cumulative_steps}")
            except (ValueError, IndexError):
                print(f"Could not extract previous steps from filename, using {cumulative_steps} as total")
        
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Creating new model instead...")
            model_path = None  # Force creation of new model
    
    if not model_path or not os.path.exists(model_path):
        # Create new model based on algorithm
        if algorithm.upper() == "PPO":
            model = PPO(
                'MlpPolicy',
                env,
                verbose=1,
                learning_rate=0.0003,
                gamma=0.99,
                tensorboard_log=log_dir
            )
        elif algorithm.upper() == "DQN":
            model = DQN(
                'MlpPolicy',
                env,
                verbose=1,
                learning_rate=0.0005,
                buffer_size=50000,
                learning_starts=1000,
                batch_size=128,
                gamma=0.99,
                exploration_fraction=0.2,
                exploration_final_eps=0.05,
                tensorboard_log=log_dir
            )
        elif algorithm.upper() == "SAC":
            model = SAC(
                'MlpPolicy',
                env,
                verbose=1,
                learning_rate=0.0003,
                buffer_size=50000,
                learning_starts=1000,
                batch_size=256,
                gamma=0.99,
                tau=0.005,
                ent_coef='auto',
                tensorboard_log=log_dir
            )
        elif algorithm.upper() == "TD3":
            model = TD3(
                'MlpPolicy',
                env,
                verbose=1,
                learning_rate=0.0003,
                buffer_size=50000,
                learning_starts=1000,
                batch_size=256,
                gamma=0.99,
                tau=0.005,
                policy_delay=2,
                tensorboard_log=log_dir
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        print(f"Training a new {algorithm} model")

    # Use the provided metrics_prefix or default to experiment_name
    if metrics_prefix is not None:
        experiment_name = experiment_name + "_" + metrics_prefix

    # Setup callback for progress tracking only
    metrics_callback = MetricsCallback(verbose=1)

    # Make a unique experiment name with timestamp to avoid _1 suffixes
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_experiment_name = f"{experiment_name}_{timestamp}"

    # Learn with callback using the unique name
    model.learn(
        total_timesteps=total_steps,
        callback=metrics_callback,
        tb_log_name=unique_experiment_name
    )

    # Save ONLY final model with consistent naming (no timestamp in the filename)
    final_model_path = f"{model_dir}{experiment_name}.zip"
    model.save(final_model_path)

    # Save ONLY final metrics with consistent naming
    final_metrics_file = f"{metrics_dir}{experiment_name}.csv"
    summary = env.save_metrics(final_metrics_file)

    # Print final stats
    print("\n######## TRAINING STATS ########")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    print(f"Final model saved to: {final_model_path}")
    print(f"Final metrics saved to: {final_metrics_file}")
    print(f"Logs saved to: {log_dir} (access with tensorboard --logdir={log_dir})")

    return env, model, summary


# Main
if __name__ == "__main__":
    env, model, summary = train_agent(
        algorithm='PPO',  # Change this to 'PPO', 'DQN', 'SAC', or 'TD3'
        mode="random",
        total_steps=100000,
        model_path='models/ppo_all_600000.zip',  # Set to existing model path if continuing training
        metrics_prefix=None
    )

# MODES:
# start = only one position + towards target
# easy = easypos + towards target
# medium = easypos + rnd direction target
# hard = hardpos + towards target
# all = hardpos + rnd direction target (all maps)
# random = all pos but maps have no limits. can select any start position and any 1-3 final positions
# random setting is used in the "no limits" map