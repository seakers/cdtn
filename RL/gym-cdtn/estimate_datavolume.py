import time
import pickle
from simulator.environments.DtnSimEnvironment import DtnSimEnviornment
from simulator.utils.DtnConfigParser import load_configuration_file, parse_configuration_dict
import numpy as np


# This script allows to calculate an estimate of the data volume per second that goes through every node in the network
# by propagating the scenario in steps of specified length, computing the number of bits that arrived at the different
# nodes for every time step and diving them by the length of the time step. The results are stored in a pickle
# "data_volume_per_second.pkl" containing a dictionary with keys being the node IDs and the values being an array of
# the estimate of the data volume per second for all simulation time steps.


def update_datavolume(environment, dv, ts):
    for nid, node in environment.nodes.items():
        data_vol = 0
        for bundle in node.endpoints[0]:
            data_vol += bundle.data_vol
        # dv[nid].append((data_vol-dv[nid][-1]) / time_step)
        dv[nid].append((data_vol - np.sum(dv[nid])) / ts)


# Define configuration file (relative to working directory)
config_file = './RL/gym-cdtn/inputs/EO_scenario_inputs/constellation_config_unconstrained.yaml'

# Load configuration file
config = load_configuration_file(config_file)

config = parse_configuration_dict(config)

# Create a simulation environment. From now on, ``config`` will be
# a global variable available to everyone
env = DtnSimEnviornment(config)

# Initialize environment, create nodes/connections, start generators
env.initialize()
t = time.time()

# Propagate with time steps
is_done = False
time_step = 1  # seconds
simulation_time = 6 * 60 * 60  # seconds

# Initialize datavolume dictionary
datavolume = {}
for nid in env.nodes:
    datavolume[nid] = [0]

while not is_done:
    current_time = env.now
    env.run(until=(current_time + time_step))
    update_datavolume(env, datavolume, time_step)
    is_done = (env.now >= simulation_time)

elapsed = time.time() - t
print("Elapsed time is " + str(elapsed) + " seconds.")

f = open("data_volume_per_second.pkl", "wb")
pickle.dump(datavolume, f)
f.close()
print("pickle saved.")
