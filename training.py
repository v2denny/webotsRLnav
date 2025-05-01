from environment import WebotsEnv
import random
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import os
import time
from datetime import datetime


# Custom callback to save metrics periodically
class MetricsCallback(BaseCallback):
    def __init__(self, save_freq=10000, metrics_path="./metrics/", experiment_name="experiment", verbose=1):
        super(MetricsCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.metrics_path = metrics_path
        self.experiment_name = experiment_name
        os.makedirs(metrics_path, exist_ok=True)
        
    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            # Save metrics to CSV with step info
            metrics_file = f"{self.metrics_path}{self.experiment_name}_step_{self.n_calls}.csv"
            self.training_env.env_method("save_metrics", metrics_file)[0]
            
            if self.verbose > 0:
                print(f"Metrics saved to {metrics_file}")
                
        return True


# Main training code
def train_agent(algorithm="PPO", mode="start", total_steps=50000, model_path=None, metrics_prefix=None):
    env = WebotsEnv()
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
    
    # Create or load model
    if model_path and os.path.exists(model_path):
        model = PPO.load(model_path, env=env, tensorboard_log=log_dir)
        print(f"Model loaded from {model_path}")
        
        # Extract previous steps from filename if possible
        try:
            # Attempt to extract steps from filename format "algo_mode_steps.zip"
            filename_parts = os.path.basename(model_path).split('_')
            if len(filename_parts) >= 3:
                prev_steps = int(filename_parts[-1].split('.')[0])
                cumulative_steps = prev_steps + total_steps
                print(f"Continuing training from step {prev_steps}, will train to {cumulative_steps}")
        except (ValueError, IndexError):
            print(f"Could not extract previous steps from filename, using {cumulative_steps} as total")
    else:
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
        print("Training a new model")
    
    # Consistent naming for all outputs
    experiment_name = f"{algorithm.lower()}_{mode}_{cumulative_steps}"
    
    # Use the provided metrics_prefix or default to experiment_name
    if metrics_prefix is None:
        metrics_prefix = experiment_name
    
    # Setup callbacks with updated naming
    checkpoint_callback = CheckpointCallback(
        save_freq=min(50000, total_steps // 2),  # Save at least twice during training
        save_path=model_dir, 
        name_prefix=experiment_name
    )
    
    metrics_callback = MetricsCallback(
        save_freq=min(10000, total_steps // 5),  # Save metrics periodically
        metrics_path=metrics_dir,
        experiment_name=metrics_prefix
    )
    
    # Learn with callbacks
    model.learn(
        total_timesteps=total_steps, 
        callback=[checkpoint_callback, metrics_callback],
        tb_log_name=experiment_name  # Specify tensorboard log name for better organization
    )
    
    # Save final model and metrics with consistent naming
    final_model_path = f"{model_dir}{experiment_name}.zip"
    model.save(final_model_path)
    
    final_metrics_file = f"{metrics_dir}{metrics_prefix}_final.csv"
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
        algorithm='PPO',
        mode="start",
        total_steps=50000,
        model_path=None,  # Set to existing model path if continuing training
        metrics_prefix=None  # Will default to "ppo_start_50000"
    )
    
    # Example of continuing training
    """
    env, model, summary = train_agent(
        algorithm='PPO',
        mode="start",
        total_steps=50000,
        model_path="models/ppo_start_50000.zip",
        metrics_prefix=None  # Will default to "ppo_start_100000"
    )
    """

# MODES:
# start = only one position + towards target
# easy = easypos + towards target
# medium = easypos + rnd direction target
# hard = hardpos + towards target
# all = hardpos + rnd direction target (all maps)
# random = all pos but maps have no limits. can select any start position and any 1-3 final positions
# random setting is used in the "no limits" map