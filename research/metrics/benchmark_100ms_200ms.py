# Improved Benchmark Plots
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the regular CSV files
df_100ms = pd.read_csv('ppo_random_300000_act3_100ms.csv')
df_200ms = pd.read_csv('ppo_random_300000_act3_200ms.csv')


# Parse summary files manually
def parse_summary_file(file_path):
    metrics = []
    values = []

    with open(file_path, 'r') as f:
        # Skip header line
        next(f)
        for line in f:
            # Split only on the first comma
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                metrics.append(parts[0])
                values.append(parts[1])

    return metrics, values


# Parse summary files
metrics_100ms, values_100ms = parse_summary_file('ppo_random_300000_act3_100ms_summary.csv')
metrics_200ms, values_200ms = parse_summary_file('ppo_random_300000_act3_200ms_summary.csv')


# Convert string values to numeric where possible
def try_convert_to_float(value_str):
    try:
        return float(value_str)
    except ValueError:
        return value_str


values_100ms_numeric = [try_convert_to_float(v) for v in values_100ms]
values_200ms_numeric = [try_convert_to_float(v) for v in values_200ms]

# 1. IMPROVED SUMMARY METRICS PLOT WITH PROPORTIONAL BARS
# ------------------------------------------------------
plt.figure(figsize=(14, 8))

# Filter out non-numeric metrics and the "actions" line
numeric_indices = []
for i, (m, v1, v2) in enumerate(zip(metrics_100ms, values_100ms_numeric, values_200ms_numeric)):
    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) and m != 'actions':
        numeric_indices.append(i)

numeric_metrics = [metrics_100ms[i] for i in numeric_indices]
numeric_values_100ms = [values_100ms_numeric[i] for i in numeric_indices]
numeric_values_200ms = [values_200ms_numeric[i] for i in numeric_indices]

# Create bar positions
x = np.arange(len(numeric_metrics))
width = 0.35

# Create figure
fig, ax = plt.subplots(figsize=(14, 7))

# Create proportional bars for each metric
for i in range(len(numeric_metrics)):
    # Calculate proportional heights (normalize to the larger of the two values)
    max_val = max(numeric_values_100ms[i], numeric_values_200ms[i])
    if max_val == 0:  # Avoid division by zero
        max_val = 1

    prop_100ms = numeric_values_100ms[i] / max_val
    prop_200ms = numeric_values_200ms[i] / max_val

    # Plot bars
    bar1 = ax.bar(x[i] - width / 2, prop_100ms, width, color='blue')
    bar2 = ax.bar(x[i] + width / 2, prop_200ms, width, color='orange')

    # Add value labels on top of bars
    ax.text(x[i] - width / 2, prop_100ms + 0.05, f'{numeric_values_100ms[i]:.2f}',
            ha='center', va='bottom', fontsize=9)
    ax.text(x[i] + width / 2, prop_200ms + 0.05, f'{numeric_values_200ms[i]:.2f}',
            ha='center', va='bottom', fontsize=9)

# Add labels and title
plt.title('Comparison of Summary Metrics: 100ms vs 200ms', fontsize=14)
plt.xticks(x, numeric_metrics, rotation=45, ha='right')

# Remove y-axis ticks
plt.yticks([])
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Add a legend
plt.figlegend(['100ms', '200ms'], loc='upper right')

# Add padding to ensure labels are visible
plt.tight_layout()
plt.savefig('improved_summary_comparison.png', dpi=300)
plt.show()

# 2. IMPROVED SMOOTHED REWARD PLOT
# --------------------------------
plt.figure(figsize=(14, 7))

# Calculate rolling average for smoother visualization
window = 50
roll_100ms = df_100ms['reward'].rolling(window=min(window, len(df_100ms))).mean()
roll_200ms = df_200ms['reward'].rolling(window=min(window, len(df_200ms))).mean()

# Only plot the rolling averages (not the raw data)
plt.plot(df_100ms['episode'], roll_100ms, 'b-', linewidth=2, label='100ms (Rolling Avg)')
plt.plot(df_200ms['episode'], roll_200ms, 'orange', linewidth=2, label='200ms (Rolling Avg)')

# Add mean lines
mean_100ms = df_100ms['reward'].mean()
mean_200ms = df_200ms['reward'].mean()
plt.axhline(y=mean_100ms, color='b', linestyle='--', alpha=0.7,
            label=f'100ms Mean: {mean_100ms:.2f}')
plt.axhline(y=mean_200ms, color='orange', linestyle='--', alpha=0.7,
            label=f'200ms Mean: {mean_200ms:.2f}')

# Set y-axis limits to focus on the relevant range
min_visible = min(roll_100ms.min(), roll_200ms.min()) * 1.1  # Add 10% padding
max_visible = max(roll_100ms.max(), roll_200ms.max()) * 1.1  # Add 10% padding
plt.ylim(min_visible, max_visible)

# Add labels and title
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Reward', fontsize=12)
plt.title('Reward vs Episode: 100ms vs 200ms (Smoothed)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig('improved_reward_comparison.png', dpi=300)
plt.show()

print("Improved benchmark plots created!")