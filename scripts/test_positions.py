import csv
import math

import time

from controller import GPS, Supervisor
from scripts.utils import warp_robot
from scripts.positions import get_positions

supervisor: Supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
gps = supervisor.getDevice('gps')
gps.enable(timestep)


def next_pos():
    init, final = get_positions("easy")
    x = float(init[0])
    y = float(init[1])
    warp_robot(supervisor, "EPUCK", (x, y))
    supervisor.step()

def print_pos():
    supervisor.step()
    gps_readings = gps.getValues()
    robot_position = (gps_readings[0], gps_readings[1])
    print(f"Robot at pos: {robot_position}")



while True:
    next_pos()
    print_pos()
    time.sleep(5)
