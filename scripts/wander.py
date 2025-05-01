import math
from numpy import random
from scripts.utils import cmd_vel
from scripts.positions import get_positions
from controller import Robot, Lidar, Supervisor

def distance_handler(direction: int, dist_values: [float]) -> (float, float):
    maxSpeed: float = 0.1
    distP: float = 10.0  # 10.0
    angleP: float = 7.0  # 7.0
    wallDist: float = 0.1

    # Find the angle of the ray that returned the minimum distance
    size: int = len(dist_values)
    min_index: int = 0
    if direction == -1:
        min_index = size - 1
    for i in range(size):
        idx: int = i
        if direction == -1:
            idx = size - 1 - i
        if dist_values[idx] < dist_values[min_index] and dist_values[idx] > 0.0:
            min_index = idx

    angle_increment: float = 2*math.pi / (size - 1)
    angleMin: float = (size // 2 - min_index) * angle_increment
    distMin: float = dist_values[min_index]
    distFront: float = dist_values[size // 2]
    distSide: float = dist_values[size // 4] if (direction == 1) else dist_values[3*size // 4]
    distBack: float = dist_values[0]

    # Prepare message for the robot's motors
    linear_vel: float
    angular_vel: float

    print("distMin", distMin)
    print("angleMin", angleMin*180/math.pi)

    # Decide the robot's behavior
    if math.isfinite(distMin):
        if distFront < 1.25*wallDist and (distSide < 1.25*wallDist or distBack < 1.25*wallDist):
            print("UNBLOCK")
            angular_vel = direction * -1
        else:
            print("REGULAR")
            angular_vel = direction * distP * (distMin - wallDist) + angleP * (angleMin - direction * math.pi / 2)
            print("angular_vel", angular_vel, " wall comp = ", direction * distP * (distMin - wallDist), ", angle comp = ", angleP * (angleMin - direction * math.pi / 2))
        if distFront < wallDist:
            # TURN
            print("TURN")
            linear_vel = 0
        elif distFront < 2 * wallDist or distMin < wallDist * 0.75 or distMin > wallDist * 1.25:
            # SLOW
            print("SLOW")
            linear_vel = 0.5 * maxSpeed
        else:
            # CRUISE
            print("CRUISE")
            linear_vel = maxSpeed
    else:
        # WANDER
        print("WANDER")
        angular_vel = random.normal(loc=0.0, scale=1.0)
        print("angular_vel", angular_vel)
        linear_vel = maxSpeed

    return linear_vel, angular_vel

if __name__ == '__main__':
    # Create the Robot instance.
    robot: Robot = Robot()

    timestep: int = int(robot.getBasicTimeStep())  # in ms

    lidar: Lidar = robot.getDevice('lidar')
    lidar.enable(timestep)
    lidar.enablePointCloud()

    init_pos, final_pos = get_positions("easy")


    # Main loop
    while robot.step() != -1:
        linear_vel, angular_vel = distance_handler(1, lidar.getRangeImage())
        cmd_vel(robot, linear_vel, angular_vel)