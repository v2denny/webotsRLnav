import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import RenderFrame
from controller import Robot, TouchSensor, DistanceSensor, Supervisor, GPS, Compass
from scripts.utils import cmd_vel
import math
from collections import Counter
from typing import List
import numpy as np
from scripts.positions import get_positions
from scripts.utils import warp_robot
import time
import os
import csv
from datetime import datetime
DIST_THRESHOLD = 0.1


class WebotsEnv(gym.Env):
    '''
    Init robot and metrics
    '''
    def __init__(self):
        # Init
        super(WebotsEnv, self).__init__()
        self.supervisor: Supervisor = Supervisor()
        timestep = int(self.supervisor.getBasicTimeStep())
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0] * 9 + [0, -np.pi]),
            high=np.array([1] * 9 + [1, np.pi]),
            dtype=np.float32)
        
        # Sensors
        self.lidar = self.supervisor.getDevice('lidar')
        self.lidar.enablePointCloud()
        self.lidar.enable(timestep)
        self.touch_sensor: TouchSensor = self.supervisor.getDevice('touch sensor')
        self.touch_sensor.enable(timestep)
        self.gps: GPS = self.supervisor.getDevice('gps')
        self.gps.enable(timestep)
        self.compass: Compass = self.supervisor.getDevice('compass')
        self.compass.enable(timestep)
        
        # Training
        self.trainmode = 'start' # Value defined in the training script
        self.robotpos, self.targetpos = get_positions(str(self.trainmode))
        print("robotpos:",self.robotpos)
        print("targetpos:",self.targetpos)
        self.targs = int(len(self.targetpos))
        print("n_targets:", self.targs)
        self.max_steps_per_episode = 1000  # Limit to prevent infinite loops
        self.prev_distance_to_target = self.calculate_distance_to_target()[0]
        self.same_spot_steps = 0

        # Track visited locations
        self.visited_locations = set()
        self.visit_threshold = 0.1  # Minimum distance to consider as a new location
        self.supervisor.step()
        self.previous_position = self.gps.getValues()[:2]

        # Metrics
        self.metrics = MetricsTracker()
        self.steps = 0

    '''
    Reset environment and metrics
    '''
    def reset(self, seed=None, options=None):
        # Set positions
        self.robotpos, self.targetpos = get_positions(str(self.trainmode))
        self.targs = int(len(self.targetpos))

        # Warp robot
        xpos, ypos = self.robotpos
        warp_robot(self.supervisor, "EPUCK", (xpos, ypos))
        self.prev_distance_to_target = self.calculate_distance_to_target()[0]
        self.steps = 0

        # Counters
        self.visited_locations = set()
        self.same_spot_steps = 0

        # Rotate robot to target
        if str(self.trainmode) in ["start", "easy", "hard"]:
            self.rotate_to_target()

        return self.get_observation(), {}

    '''
    End episode if collision is detected
    Choose and deploy action
    Truncate episode if max steps are reached
    Calculate step reward
    '''
    def step(self, action):
        self.steps += 1

        # Choose and execute the action
        if action == 0:
            cmd_vel(self.supervisor, 0.12, 0)
        elif action == 1:
            cmd_vel(self.supervisor, 0, 1.3)
        elif action == 2:
            cmd_vel(self.supervisor, 0, -1.3)
        self.supervisor.step(100)

        self.metrics.log_action(action)

        # Check for collisions
        done = self.collision()
        if done == True :
            self.metrics.log_step(-100)
            self.metrics.log_episode(False, True, False, len(self.visited_locations))
            return self.get_observation(), -100, True, False, {}

        truncated = self.steps >= self.max_steps_per_episode
        reward, done, success = self.calculate_reward()

        # Check for truncation
        if truncated:
            self.metrics.log_step(0)
            self.metrics.log_episode(False, False, True, len(self.visited_locations))
            return self.get_observation(), reward, done, truncated, {}

        # Check for success
        if success:
            self.metrics.log_step(reward)
            self.metrics.log_episode(True, False, False, len(self.visited_locations))
            return self.get_observation(), reward, done, truncated, {}

        # Log normal step
        self.metrics.log_step(reward)

        return self.get_observation(), reward, done, truncated, {}

    '''
    Detect collision using touch sensor readings
    '''
    def collision(self):
        self.supervisor.step()
        if self.touch_sensor.getValue() == 1.0:
            return True
        else: return False

    '''
    Rotate the robot towards the target
    Function used in the beginning of episodes
    '''
    def rotate_to_target(self):
        obs = self.get_observation()
        ang = float(obs[-1])
        while abs(ang) >= 0.1:
            cmd_vel(self.supervisor, 0, 2)
            self.supervisor.step()
            obs = self.get_observation()
            ang = float(obs[-1])
        cmd_vel(self.supervisor, 0, 0)
        self.supervisor.step(1)

    '''
    Observations are:
    - Lidar features (9 rays)
    - Distance to target
    - Angle towards target
    '''
    def get_observation(self):
        self.supervisor.step()
        point_cloud = np.array(self.lidar.getPointCloud())
        lidar_features = self.process_lidar(point_cloud)
        distance, nearest_target = self.calculate_distance_to_target()
        angle = self.calculate_angle_to_target(nearest_target)
        observations = np.concatenate([lidar_features, [distance, angle]])
        return observations
    
    '''
    Reward function:
    - Penalty for collision
    - Reward for reaching targets
    - Distance-based penalty
    - Getting closer reward
    - Time penalty
    - Exploration reward
    - Movement reward
    - Close to wall penalty
    - Anti-spin penalty
    '''
    def calculate_reward(self):
        self.supervisor.step()
        distance, closest_target = self.calculate_distance_to_target()
        success = False

        # Collision penalty
        if self.touch_sensor.getValue() == 1.0:
            done = True
            reward = -100
            return reward, done, success

        # Reached the target
        if distance < DIST_THRESHOLD and self.targs == 1:
            self.targs -= 1
            done = True
            reward = 100
            success = True
        elif distance < DIST_THRESHOLD and self.targs > 1:
            self.targs = self.targs - 1
            done = False
            reward = 80
            self.targetpos.remove(closest_target)
        else:
            done = False
            reward = -distance*0.5  # Negative reward based on distance to the target

        # Additional reward shaping
        if distance <= self.prev_distance_to_target: reward += 5 * (self.prev_distance_to_target - distance)  # Reward for getting closer
        reward -= 0.8  # Small negative reward for each step taken (time penalty)

        # Check if the current position is a new location
        gps_readings: List[float] = self.gps.getValues()
        robot_position = (gps_readings[0], gps_readings[1])
        current_position = (round(robot_position[0], 1), round(robot_position[1], 1))
        if current_position not in self.visited_locations:
            self.visited_locations.add(current_position)
            reward += 2  # Reward for visiting a new location

        distance_moved = np.linalg.norm(np.array(robot_position) - np.array(self.previous_position))
        reward += distance_moved * 2  # Reward for movement

        # Proximity to wall penalty
        point_cloud = np.array(self.lidar.getPointCloud())
        lidar_features = self.process_lidar(point_cloud)
        for feature in lidar_features:
            if feature <= 0.02: reward -= 1

        # Penalty for staying in the same spot (spinning)
        movement_magnitude = np.linalg.norm(np.array(robot_position) - np.array(self.previous_position))

        if movement_magnitude < 0.005:
            self.same_spot_steps += 1
        else:
            self.same_spot_steps = 0

        if self.same_spot_steps > 10:
            reward -= 10

        # Update previous values
        self.prev_distance_to_target = distance
        self.previous_position = robot_position

        return reward, done, success

    '''
    Calculate distance to 'main' target
    '''
    def calculate_distance_to_target(self):
        dismin = 100000
        mintargetpos = None
        for targetpos in self.targetpos:
            (xtarget, ytarget) = targetpos
            self.supervisor.step()
            gps_readings: List[float] = self.gps.getValues()
            robot_position = (gps_readings[0], gps_readings[1])
            distance = math.sqrt((robot_position[0] - xtarget) ** 2 + (robot_position[1] - ytarget) ** 2)
            if distance < dismin:
                dismin = distance
                mintargetpos = targetpos
        return dismin, mintargetpos
    
    '''
    Calculate the orientation angle of the robot
    '''
    def get_robot_orientation(self) -> float:
        self.supervisor.step()
        compass_values = self.compass.getValues()
        orientation = math.atan2(compass_values[0], compass_values[1])
        return orientation

    '''
    Get current GPS readings and robot orientation
    '''
    def calculate_angle_to_target(self, target_pos) -> float:
        self.supervisor.step()
        gps_readings: List[float] = self.gps.getValues()
        robot_position = (gps_readings[0], gps_readings[1])
        robot_orientation = self.get_robot_orientation()

        target_vector = (target_pos[0] - robot_position[0], target_pos[1] - robot_position[1])
        angle_to_target = math.atan2(target_vector[1], target_vector[0])
        relative_angle = angle_to_target - robot_orientation
        relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi

        return relative_angle

    '''
    Get 9 equidistant rays of lidar
    Keeps the observation space simple
    Enough rays for this task
    '''
    def process_lidar(self, point_cloud):
        distances = []

        # Select 9 points from the lidar readings (100 total)
        ids = [0, 11, 24, 36, 49, 61, 74, 86, 99]
        pc9 = [point_cloud[i] for i in ids]

        for point in pc9:
            x = point.x
            y = point.y
            distance = np.sqrt(x ** 2 + y ** 2)
            distances.append(min(distance, 2))

        # Normalize to range [0, 1]
        distances = np.array(distances)
        distances = np.clip(distances / 2, 0, 1)
        return distances

    '''
    Save current metrics to CSV file
    '''
    def save_metrics(self, filename="training_metrics.csv", append=False):
        self.metrics.save_to_csv(filename, append)
        return self.metrics.get_summary()

    def render(self, mode='human'):
        pass



'''
Metric tracking and logging class
'''
class MetricsTracker:
    def __init__(self):
        # Training-level metrics
        self.episode_rewards = []
        self.unique_locations_count = []
        self.episode_steps = []
        self.episode_successes = []
        self.start_time = time.time()
        self.total_episodes = 0
        self.actions = []
        self.successful_episodes = 0
        self.total_collisions = 0
        self.total_truncations = 0
        
        # Episode-level metrics (reset each episode)
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        self.done = False
        self.collision = False
        self.truncation = False
        
    def reset_episode_metrics(self):
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        self.done = False
        self.collision = False
        self.truncation = False

    '''
    Log metrics every step
    '''
    def log_step(self, reward):
        self.current_episode_reward += reward
        self.current_episode_steps += 1

    def log_action(self, action):
        self.actions.append(action)

    '''
    Log metrics at end of each episode
    '''
    def log_episode(self, success, collision, truncation, unique_locations_count):
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_steps.append(self.current_episode_steps)
        self.unique_locations_count.append(unique_locations_count)
        self.total_episodes += 1
        if success:
            self.successful_episodes += 1
            self.episode_successes.append('Success')
        if collision:
            self.total_collisions += 1
            self.episode_successes.append('Collision')
        if truncation:
            self.total_truncations += 1
            self.episode_successes.append('Truncation')

        self.reset_episode_metrics()

    def calculate_percentages(self, values):
        counter = Counter(values)
        total = sum(counter.values())
        percentages = {val: (count / total) * 100 for val, count in counter.items()}
        result = ', '.join([f"{val}-{int(percentage)}%" for val, percentage in sorted(percentages.items())])
        return result

    def get_summary(self):
        training_time = time.time() - self.start_time

        return {
            "total_episodes": self.total_episodes,
            "successful_episodes": self.successful_episodes,
            "success_rate": (self.successful_episodes / self.total_episodes) * 100,
            "collision_rate": (self.total_collisions / self.total_episodes) * 100,
            "truncation_rate": (self.total_truncations / self.total_episodes) * 100,
            "actions": self.calculate_percentages(self.actions),
            "avg_episode_steps": sum(self.episode_steps) / self.total_episodes,
            "avg_episode_reward": sum(self.episode_rewards) / self.total_episodes,
            "avg_unique_locations": sum(self.unique_locations_count) / max(1, len(self.unique_locations_count)),
            "total_steps": sum(self.episode_steps),
            "training_time_seconds": training_time
        }
    
    def save_to_csv(self, filename="training_metrics.csv", append=False):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Episode-level data
        episode_data = {
            "episode": list(range(1, self.total_episodes + 1)),
            "reward": self.episode_rewards,
            "steps": self.episode_steps,
            "unique_locations": self.unique_locations_count,
            "success": self.episode_successes
        }
        
        # Save episode-level data
        mode = 'a' if append else 'w'
        with open(filename, mode, newline='') as f:
            if not append or os.path.getsize(filename) == 0:
                # Write header
                header = ','.join(episode_data.keys())
                f.write(header + '\n')
                
            # Write data rows
            for i in range(len(episode_data["episode"])):
                row = [str(episode_data[key][i]) for key in episode_data.keys()]
                f.write(','.join(row) + '\n')
        
        # Save summary data to a separate file
        summary = self.get_summary()
        summary_filename = filename.replace('.csv', '_summary.csv')
        with open(summary_filename, 'w', newline='') as f:
            # Write header and data as key,value pairs
            f.write('metric,value\n')
            for key, value in summary.items():
                f.write(f"{key},{value}\n")