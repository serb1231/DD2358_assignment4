import numpy as np
import matplotlib.pyplot as plt
import random

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

def simulate_wildfire(i = 0):
    """Simulates wildfire spread over time."""
    forest, burn_time = initialize_forest()
    
    fire_spread = []  # Track number of burning trees each day
    ash_spread = []
    for day in range(DAYS):
        new_forest = forest.copy()
        
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if forest[x, y] == BURNING:
                    burn_time[x, y] += 1  # Increase burn time
                    
                    # If burn time exceeds threshold, turn to ash
                    if burn_time[x, y] >= BURN_TIME:
                        new_forest[x, y] = ASH
                    
                    # Spread fire to neighbors
                    for nx, ny in get_neighbors(x, y):
                        if forest[nx, ny] == TREE and random.random() < FIRE_SPREAD_PROB:
                            new_forest[nx, ny] = BURNING
                            burn_time[nx, ny] = 1
        
        forest = new_forest.copy()
        fire_spread.append(np.sum(forest == BURNING))
        ash_spread.append(np.sum(forest == ASH))
        
        if np.sum(forest == BURNING) == 0:  # Stop if no more fire
            break
        
    return fire_spread, ash_spread

# # Run simulation
# fire_spread_over_time, ash_spread_over_time = simulate_wildfire()

# print("Poor trees burnt at every step", fire_spread_over_time)
# print("Poor trees that are now just ash", ash_spread_over_time)

import multiprocessing
import time

def main():
    num_workers = 15
    start_time = time.time()
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(simulate_wildfire, [1] * num_workers * 10)
    end_time = time.time()
    total_time_taken = end_time - start_time
    # compute average wildfire spread by taking the last value of the ash and burnt arrays and sum them up
    total_wildfire_spread = 0

    for result in results:
        total_wildfire_spread += result[0][-1] + result[1][-1]

    print("average catastrophic result: ", total_wildfire_spread / num_workers)
    print("total time taken by multiproc: ", total_time_taken)

if __name__ == "__main__":
    main()
