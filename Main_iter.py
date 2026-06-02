import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # local code dir (was C:\Users)
_HERE = os.path.dirname(os.path.abspath(__file__))

import multiprocessing
import time
import numpy as np
import pandas as pd
import os

#from Functions_code_RRT import *  # For running RRT simulation
from Functions_code import *     # For running hybrid algorithm

from agent_navigation import agent_run_simulation

from multiprocessing import Lock
lock=Lock()

def monitor_task_counts(task_counts, pause_flags, threshold=200, interval=1): 
    """
    Monitor task counts every interval seconds and deprioritize agents significantly ahead.
    """
    while True:
        time.sleep(interval)  # Wait for the next check

        # Find the max and min task counts
        max_tasks = max(task_counts.values())
        min_tasks = min(task_counts.values())
        
        # Identify agents to pause
        for agent_id, count in task_counts.items():
            if  count - min_tasks > threshold:
                pause_flags[agent_id] = True  # Pause the agent

            else:
                pause_flags[agent_id] = False  # Resume the agent


def agent_worker(agent_id, agent_states, shared_stats, shared_env, stop_signal, position_array, task_counts, pause_flags):
    """
    Worker function for each agent.
    """
    try:
        agent_run_simulation(agent_id, agent_states, shared_stats, shared_env, stop_signal, position_array, task_counts, pause_flags)
    except Exception:
        import traceback
        print(f"!!! Agent {agent_id} CRASHED:", flush=True)
        traceback.print_exc()

def main(routes, up_inflated_matrix, original_scaled_matrix, original_black_ranges, time_limit_minutes, num_agents, max_iterations=3010):
    """
    Main function to manage agents and coordinate the simulation.
    """
    # Convert time limit to seconds
    time_limit = time_limit_minutes * 60
    manager = multiprocessing.Manager()
    stop_signal = multiprocessing.Event()  # Signal to stop agents
    # max_iterations is now a parameter (was hard-coded 3010)

    task_counts = manager.dict({i: 0 for i in range(num_agents)})  # Initialize task counts
    pause_flags = manager.dict({i: False for i in range(num_agents)})  # Initialize pause flags

    # Create shared position array - keep it separate from shared_env
    position_array = multiprocessing.Array('d', [0.0] * (num_agents * 2))  # x,y coordinates for each agent

    # Shared statistics
    shared_stats = manager.dict({
        'COUNT_ALL_COLL': 0,
        'COUNT_ALL_FAILED': 0,
        'COUNT_ALL_ROUTES': 0,
        'TOTAL_TASKS_COMPLETED': 0,
        'COLL_STATS': {50: None, 100: None, 300: None, 500: None, 700: None, 1000: None, 3000: None},  # Dictionary to track collision counts at milestones,

    })

    # Shared environment
    shared_env = manager.dict({
        'original_scaled_matrix': original_scaled_matrix.tolist() if isinstance(original_scaled_matrix, np.ndarray) else original_scaled_matrix,
        'up_inflated_matrix': up_inflated_matrix.tolist() if isinstance(up_inflated_matrix, np.ndarray) else up_inflated_matrix,
        'original_black_ranges': original_black_ranges,
        'routes': routes,
        'total_collision_grid':np.zeros_like(up_inflated_matrix),
    })

    # Shared agent states
    agent_states = manager.dict({
        i: {
            "current_position": None,
            "tasks": [],
            "completed_tasks": 0,
            "collisions": 0,
            "failed_tasks": 0,
        }
        for i in range(num_agents)
    })

    # Generate start and end points for initial tasks
    print("main(): generating start/end points...", flush=True)
    start_points, end_points = generate_start_and_end(up_inflated_matrix, original_black_ranges, num_agents=num_agents, min_distance=4)
    print(f"main(): start/end ready, spawning {num_agents} agents...", flush=True)

    for agent_id in range(num_agents):
        # Fetch the current agent state
        agent_state = agent_states[agent_id]  # Retrieve the dictionary for this agent

        # Update the agent's state
        agent_state["current_position"] = start_points[agent_id]
        agent_state["tasks"] = [{"start_point": start_points[agent_id], "end_point": end_points[agent_id]}]

        # Reassign the modified state back to agent_states
        agent_states[agent_id] = agent_state

        # Initialize position array separately
        position_array[agent_id * 2] = start_points[agent_id][0]
        position_array[agent_id * 2 + 1] = start_points[agent_id][1]

    # Create and start processes for agents
    processes = []
    for agent_id in range(num_agents):
        process = multiprocessing.Process(
            target=agent_worker,
            args=(agent_id, agent_states, shared_stats, shared_env, stop_signal, position_array, task_counts, pause_flags)
        )
        processes.append(process)
        process.start()

    # Start the monitor thread
    monitor_thread = multiprocessing.Process(target=monitor_task_counts, args=(task_counts, pause_flags))
    monitor_thread.daemon = True  # Ensures the thread stops when the main program exits
    monitor_thread.start()


    # Monitor iterations
    _t0 = time.time(); _last = 0.0
    while True:
        completed_iterations = shared_stats["TOTAL_TASKS_COMPLETED"] - shared_stats["COUNT_ALL_FAILED"]

        if time.time() - _last > 5:
            print(f"[hb t={time.time()-_t0:.0f}s] net={completed_iterations} "
                  f"total={shared_stats['TOTAL_TASKS_COMPLETED']} failed={shared_stats['COUNT_ALL_FAILED']} "
                  f"coll={shared_stats['COUNT_ALL_COLL']}", flush=True)
            _last = time.time()

        with lock:
            coll_stats = shared_stats["COLL_STATS"]  # Create a local copy
            current_collisions = shared_stats["COUNT_ALL_COLL"]

            for milestone in coll_stats.keys():
                if completed_iterations >= milestone and coll_stats[milestone] is None:
                    coll_stats[milestone] = current_collisions

            shared_stats["COLL_STATS"] = coll_stats  # Write the updated copy back

        if completed_iterations >= max_iterations:
            stop_signal.set()
            break
        time.sleep(0.1)  # Allow agents to work


    # Stop all agents
    stop_signal.set()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    # Print global statistics
    keys_to_print = ['COUNT_ALL_COLL', 'COUNT_ALL_FAILED', 'COUNT_ALL_ROUTES', 'TOTAL_TASKS_COMPLETED']
    filtered_stats = {key: shared_stats[key] for key in keys_to_print}
    print(f"Global Statistics of {num_agents} agents: {filtered_stats}")
    print("total mission: ", shared_stats['TOTAL_TASKS_COMPLETED']- shared_stats['COUNT_ALL_FAILED'])

    # Print the COLL_STATS dictionary
    print("\nCollision Milestones (COLL_STATS):")
    for milestone, collisions in shared_stats['COLL_STATS'].items():
        print(f"Milestone {milestone}: {collisions} collisions")

    np.savetxt(os.path.join(_HERE, 'last_collision_grid.txt'), shared_env["total_collision_grid"], fmt='%d')
    return {
        'num_agents': num_agents,
        'collisions': shared_stats['COUNT_ALL_COLL'],
        'failed': shared_stats['COUNT_ALL_FAILED'],
        'routes': shared_stats['COUNT_ALL_ROUTES'],
        'tasks_completed': shared_stats['TOTAL_TASKS_COMPLETED'],
        'net_missions': shared_stats['TOTAL_TASKS_COMPLETED'] - shared_stats['COUNT_ALL_FAILED'],
    }



if __name__ == '__main__':


    # for map 1:
    original_scaled_matrix_1 = np.loadtxt(os.path.join(_HERE, 'env', 'map1.txt'), delimiter='\t')

    original_black_ranges_1 = [
        [(0, 50)], # y= 0 is the index of the list
        [(0, 50)], # y = 1
        [(0, 50)], # y = 2
        [(0, 50)], #y = 3
        [(0, 3), (15, 25), (47, 50)], # y = 4
        [(0, 3), (15, 25),(33, 37), (47, 50)],# y = 5
        [(0, 3), (15, 25),(33, 37), (47, 50)], # y = 6
        [(0, 3), (7, 11), (15, 25), (33, 41), (47, 50)],# y = 7
        [(0, 3), (7, 11), (21, 25), (33, 41), (47, 50)], # y = 8
        [(0, 3), (7, 11), (21, 25), (33, 41), (47, 50)], # y = 9
        [(0, 3), (7, 11), (21, 25), (33, 41), (47, 50)], # y = 10
        [(0, 3), (7, 19), (21, 25), (27, 41), (47, 50)], # y = 11
        [(0, 3), (7, 19), (21, 25), (27, 39), (47, 50)], # y = 12
        [(0, 3), (7, 19), (21, 25), (27, 39), (47, 50)], # y = 13
        [(0, 3), (7, 19), (21, 25), (27, 39), (47, 50)], # y = 14
        [(0, 3), (7,19),(21,25),(27,39),(47, 50)],# y = 15
        [(0, 3), (35, 39), (47, 50)],# y = 16
        [(0, 3), (35, 39), (47, 50)],# y = 17
        [(0, 3), (35, 39), (47, 50)],# y = 18
        [(0, 3), (35, 45), (47, 50)],# y = 19
        [(0, 3), (35, 45), (47, 50)],# y = 20
        [(0, 3), (7, 11), (35, 45), (47, 50)],# y = 21
        [(0, 3), (7, 11), (35, 45), (47, 50)],# y = 22
        [(0, 3), (7, 11), (23, 27), (35, 45), (47, 50)],# y = 23
        [(0, 3), (7, 11), (23, 27), (35, 39), (47, 50)],# y = 24
        [(0, 3), (7, 15), (23, 27), (35, 39), (47, 50)],# y = 25
        [(0, 3), (7, 15), (23, 27), (35, 39), (47, 50)],# y = 26
        [(0, 3), (7, 15), (23, 27),(35,39),(47, 50)],# y = 27
        [(0, 3), (7, 15), (23, 27), (47, 50)],# y = 28
        [(0, 3), (7, 15),(23, 27), (47, 50)],# y = 29
        [(0, 3),(11, 15), (23, 27), (47, 50)],# y = 30
        [(0, 3),(11,15),(23,27),(45, 50)],# y = 31
        [(0, 3), (45, 50)],# y = 32
        [(0, 3),(11, 15), (23, 27), (45, 50)],# y = 33
        [(0, 3),(11, 15), (23, 27), (45, 50)],# y = 34
        [(0, 15), (23, 39), (45, 50)],# y = 35
        [(0, 15), (23, 39), (47, 50)],# y = 36
        [(0, 15), (23, 39), (47, 50)],# y = 37
        [(0, 15), (23, 39), (47, 50)],# y = 38
        [(0, 15), (23, 39), (47, 50)],# y = 39
        [(0, 3),(23, 27), (47, 50)],# y = 40
        [(0, 3),(23,27), (33, 41), (47, 50)],# y = 41
        [(0, 3), (33, 41), (47, 50)],# y = 42
        [(0, 3), (23, 27), (33, 41), (47, 50)],# y = 43
        [(0, 3), (23, 27), (33, 41), (47, 50)],# y = 44
        [(0, 3), (23, 27), (33, 41), (47, 50)],# y = 45
        [(0, 3), (23, 27), (33, 41), (47, 50)],# y = 46
        [(0, 50)],# y = 47
        [(0, 50)],# y = 48
        [(0, 50)],# y = 49
        [(0, 50)]# y = 50
    ]


    # for map 2:
    original_scaled_matrix_2 = np.loadtxt(os.path.join(_HERE, 'env', 'map2.txt'), delimiter='\t')
    
    original_black_ranges_2 = [
        [(0, 50)], # y= 0 is the index of the list
        [(0, 50)], # y = 1
        [(0, 50)], # y = 2
        [(0, 50)], #y = 3
        [(0, 3), (16, 25), (47, 50)], # y = 4
        [(0, 3), (16, 25), (47, 50)],# y = 5
        [(0, 3), (16, 25), (47, 50)], # y = 6
        [(0, 3), (7, 11), (16, 25), (33, 41), (47, 50)],# y = 7
        [(0, 3), (7, 11), (21, 25), (33, 41), (47, 50)], # y = 8
        [(0, 3), (7, 11), (21, 25), (33, 41), (47, 50)], # y = 9
        [(0, 3), (7, 11), (21, 25), (33, 41), (47, 50)], # y = 10
        [(0, 3), (7, 16), (21, 25), (28, 41), (47, 50)], # y = 11
        [(0, 3), (7, 16), (21, 25), (28, 39), (47, 50)], # y = 12
        [(0, 3), (7, 16), (21, 25), (28, 39), (45, 50)], # y = 13
        [(0, 3), (7, 16), (21, 25), (28, 39), (45, 50)], # y = 14
        [(0, 3), (7, 16), (21,25),(27,39),(45, 50)],# y = 15
        [(0, 3), (7, 11), (32, 39), (45, 50)],# y = 16
        [(0, 3), (7, 11), (35, 40), (47, 50)],# y = 17
        [(0, 3), (17, 20), (35, 40), (47, 50)],# y = 18
        [(0, 3), (17, 20), (35, 43), (47, 50)],# y = 19
        [(0, 3), (17, 20), (35, 43), (47, 50)],# y = 20
        [(0, 3), (17, 20), (35, 43), (47, 50)],# y = 21
        [(0, 3), (7, 11), (35, 43), (47, 50)],# y = 22
        [(0, 3), (7, 11), (23, 27), (35, 39), (47, 50)],# y = 23
        [(0, 3), (7, 11), (23, 27), (35, 39), (47, 50)],# y = 24
        [(0, 3), (7, 14), (22, 29), (35, 39), (47, 50)],# y = 25
        [(0, 3), (7, 14), (17, 20), (22, 29), (47, 50)],# y = 26
        [(0, 3), (7, 14), (17, 20), (22, 29), (47, 50)],# y = 27
        [(0, 3), (7, 14), (17, 20), (22, 29), (47, 50)],# y = 28
        [(0, 3), (7, 14), (17, 20), (22, 29), (47, 50)],# y = 29
        [(0, 3), (47, 50)],# y = 30
        [(0, 3), (45, 50)],# y = 31
        [(0, 3), (45, 50)],# y = 32
        [(0, 3),(11, 14), (23, 27), (34, 37), (45, 50)],# y = 33
        [(0, 3),(11, 14), (17,20), (23, 27), (34, 37), (45, 50)],# y = 34
        [(0, 14), (17, 20), (23, 38), (45, 50)],# y = 35
        [(0, 14), (17, 20), (23, 38), (47, 50)],# y = 36
        [(0, 13), (17, 20), (23, 38), (47, 50)],# y = 37
        [(0, 13), (17, 20), (23, 38), (47, 50)],# y = 38
        [(0, 13), (17, 20), (23, 38), (47, 50)],# y = 39
        [(0, 3), (17, 20), (23, 27), (47, 50)],# y = 40
        [(0, 3), (23, 27), (47, 50)],# y = 41
        [(0, 3), (33, 37), (47, 50)],# y = 42
        [(0, 3), (7, 10), (33, 41), (47, 50)],# y = 43
        [(0, 3), (7, 10), (20, 23), (33, 41), (47, 50)],# y = 44
        [(0, 3), (7, 10), (20, 27), (33, 41), (47, 50)],# y = 45
        [(0, 3), (7, 10), (20, 27), (33, 41), (47, 50)],# y = 46
        [(0, 50)],# y = 47
        [(0, 50)],# y = 48
        [(0, 50)],# y = 49
        [(0, 50)]# y = 50
    ]


    # Alternative:
    # Load the alternative\original dataset
    file_path = os.path.join(_HERE, 'dataset', 'alternative_data.csv')

    df = pd.read_csv(file_path)
    # Remove rows with NaN values and invalid ranges
    df = df.dropna()

    # Extract complete routes
    routes = extract_routes_newdata(df)

    # choose Map:
    original_scaled_matrix, original_black_ranges = original_scaled_matrix_2, original_black_ranges_2
    _, up_inflated_matrix, up_black_ranges = Update1(original_scaled_matrix, original_black_ranges, [[0,0]])

    # Filter routes
    print(f"PREP: filtering {len(routes)} routes...", flush=True); _tf=time.time()
    filtered_routes = filter_routes_through_black_ranges_and_distance(routes, up_black_ranges)
    routes = filtered_routes
    print(f"PREP: routes after filter = {len(routes)} ({time.time()-_tf:.1f}s)", flush=True)

    '''
    # Original:
    # Load the original dataset 
    file_path = "C:\\Users\\original_data.csv" # add path for original dataset
    df = pd.read_csv(file_path)
    # Remove rows with NaN values and invalid ranges
    df = df.dropna()
    df = df[~((df.iloc[:, -2:] < -2) | (df.iloc[:, -2:] > 2)).any(axis=1)]
    # Extract only the first two columns
    df = df.iloc[:, :2]

    # Extract complete routes
    routes = extract_routes(df)
    print("Number of routes extracted: ", len(routes))

    # choose Map:
    original_scaled_matrix, original_black_ranges = original_scaled_matrix_1, original_black_ranges_1
    _, up_inflated_matrix, up_black_ranges = Update1(original_scaled_matrix, original_black_ranges, [[0,0]])

    filtered_routes = filter_routes_through_black_ranges_and_distance(routes, up_black_ranges)  #filter routes on map1 and after that choose map2 
    routes= filtered_routes
    print("Number of routes extracted after filter: ", len(routes))

    # choose Map:
    original_scaled_matrix, original_black_ranges = original_scaled_matrix_2, original_black_ranges_2
    _, up_inflated_matrix, up_black_ranges = Update22(original_scaled_matrix, original_black_ranges, [[0,0]])
    '''



    # START SIMULATION
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument('--agents', type=int, default=5)
    _ap.add_argument('--iters', type=int, default=3010)
    _args, _ = _ap.parse_known_args()
    time_limit_minutes = 99999  # iteration-limited (time cap effectively disabled)
    stats = main(routes, up_inflated_matrix, original_scaled_matrix, original_black_ranges,
                 time_limit_minutes, _args.agents, max_iterations=_args.iters)
    print('RESULT_JSON:', stats, flush=True)
