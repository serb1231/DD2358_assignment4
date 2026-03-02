import numpy as np
import matplotlib.pyplot as plt
import random
from dask import delayed
import dask
import dask.array as da
from dask.distributed import Client

# Constants
GRID_SIZE = 800  # 800x800 forest grid
FIRE_SPREAD_PROB = 0.3  # Probability that fire spreads to a neighboring tree
BURN_TIME = 3  # Time before a tree turns into ash
DAYS = 60  # Maximum simulation time

# State definitions
EMPTY = 0    # No tree
TREE = 1     # Healthy tree 
BURNING = 2  # Burning tree 
ASH = 3      # Burned tree 

def initialize_forest():
    """Creates a forest grid with all trees and ignites one random tree."""
    forest = np.ones((GRID_SIZE, GRID_SIZE), dtype=int)  # All trees
    burn_time = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)  # Tracks how long a tree burns
    
    # Ignite a random tree
    x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    forest[x, y] = BURNING
    burn_time[x, y] = 1  # Fire starts burning
    
    return forest, burn_time

def get_neighbors(x, y):
    """Returns the neighboring coordinates of a cell in the grid."""
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            neighbors.append((nx, ny))
    return neighbors

# import numpy as np
import dask.array as da
@delayed
def simulate_wildfire(sim_id=0):
    # Use int8 to save memory; GRID_SIZE 800 is large!
    forest = np.full((GRID_SIZE, GRID_SIZE), TREE, dtype=np.int8)
    burn_time = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    
    x = np.random.randint(0, GRID_SIZE)
    y = np.random.randint(0, GRID_SIZE)
    forest[x, y] = BURNING
    burn_time[x, y] = 1
    
    fire_history = []
    ash_history = []

    for day in range(DAYS):

        burning_mask = (forest == BURNING)
        
        # get the values to left right up and down
        north = np.zeros_like(burning_mask); north[:-1, :] = burning_mask[1:, :]
        south = np.zeros_like(burning_mask); south[1:, :] = burning_mask[:-1, :]
        east  = np.zeros_like(burning_mask); east[:, 1:] = burning_mask[:, :-1]
        west  = np.zeros_like(burning_mask); west[:, :-1] = burning_mask[:, 1:]
        
        neighbors_at_risk = north | south | east | west
        
        # so we will use some masks in order to get the right data
        luck_roll = np.random.random((GRID_SIZE, GRID_SIZE)) < FIRE_SPREAD_PROB
        new_ignitions = neighbors_at_risk & (forest == TREE) & luck_roll
        
        # increase the time for becoming ash
        burn_time[burning_mask] += 1
        
        # use the mask as well as the burning time
        become_ash = burning_mask & (burn_time >= BURN_TIME)
        
        forest[new_ignitions] = BURNING
        burn_time[new_ignitions] = 1
        
        forest[become_ash] = ASH

        # Stats tracking
        current_burning = np.sum(forest == BURNING)
        fire_history.append(current_burning)
        ash_history.append(np.sum(forest == ASH))

        if current_burning == 0:
            break

    return fire_history, ash_history

import multiprocessing
import time

def main():
    client = Client(n_workers=15, threads_per_worker=1) 
    print(f"Dashboard link: {client.dashboard_link}")
    num_workers = 1500
    start_time = time.time()
    simulations = [simulate_wildfire() for _ in range(num_workers)]


    @delayed
    def get_impact(res):
        return res[0][-1] + res[1][-1]

    impact_tasks = [get_impact(s) for s in simulations]

    impact_da = da.stack([da.from_delayed(t, shape=(), dtype=int) for t in impact_tasks])

    avg_impact_task = impact_da.mean()

    final_res = avg_impact_task.compute()

    end_time = time.time()
    total_time_taken = end_time - start_time
    # compute average wildfire spread by taking the last value of the ash and burnt arrays and sum them up
    total_wildfire_spread = 0

    # for result in results:
    #     total_wildfire_spread += result[0][-1] + result[1][-1]

    print("average catastrophic result: ", final_res)
    print("total time taken by multiproc: ", total_time_taken)
    client.close()
if __name__ == "__main__":
    main()
