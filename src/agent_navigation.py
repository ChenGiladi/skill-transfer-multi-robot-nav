import sys
sys.path.append("C:\\Users") # add path


import time
import numpy as np

#from Functions_code_RRT import *  # For running RRT simulation
from Functions_code import *     # For running hybrid algorithm (original and alternative data sets)

from multiprocessing import Lock
lock=Lock()


def agent_run_simulation(agent_id, agent_states, shared_stats, shared_env, stop_signal, position_array, task_counts, pause_flags):


    agent_state = agent_states[agent_id]
    
    # Initialize agent-specific data
    current_position = np.array(agent_state["current_position"])
    tasks = agent_state["tasks"]
    completed_tasks = agent_state["completed_tasks"]
    collisions = agent_state["collisions"]
    failed_tasks = agent_state["failed_tasks"]
    
    # Extract shared environment matrices
    original_scaled_matrix = np.array(shared_env['original_scaled_matrix'])
    up_inflated_matrix = np.array(shared_env['up_inflated_matrix'])
    original_black_ranges = shared_env['original_black_ranges']
    up_black_ranges = (shared_env['original_black_ranges']).copy() 
    routes = shared_env['routes']
    
    # Update initial position in shared array
    position_array[agent_id * 2] = current_position[0]
    position_array[agent_id * 2 + 1] = current_position[1]

    # Main loop - runs until time runs out
    while not stop_signal.is_set():

        # Check if this agent is paused
        if pause_flags[agent_id]:
            time.sleep(2)  # Wait while paused
            continue

        if not tasks:
            # Generate new task when task list is empty
            new_start, new_end = generate_start_and_end(
                up_inflated_matrix, up_black_ranges, num_agents=1, min_distance=4
            )
            tasks.append({"start_point": new_start[0], "end_point": new_end[0]})
            
        # Get new task and set initial positions
        task = tasks.pop(0)
        start_point, end_point = (task["start_point"]), (task["end_point"])
        
        # Initialize position for new task
        agent_state["current_position"] = start_point
        agent_state["tasks"] = [{"start_point": start_point, "end_point": end_point}]
        current_position = agent_state["current_position"]

        position_array[agent_id * 2] = current_position[0]
        position_array[agent_id * 2 + 1] = current_position[1]

                
        agent_collision_grid=np.zeros_like(up_inflated_matrix)
        path_taken = []
        path_taken = [current_position]
        best_route = []
        success = False
        k_value = 0
        step_count = 0
        MAX_TASK_STEPS = 400  # safeguard: a non-converging (stuck) task is counted as failed

        # Second loop - runs until current task is completed or failed
        while not success:
            if stop_signal.is_set():
                break
            step_count += 1
            if step_count > MAX_TASK_STEPS:
                with lock:
                    shared_stats["COUNT_ALL_FAILED"] += 1
                    shared_stats["TOTAL_TASKS_COMPLETED"] += 1
                failed_tasks += 1
                break

            obstacles = []
            for i in range(len(agent_states)):
                if i != agent_id:
                    obs_x = position_array[i * 2]
                    obs_y = position_array[i * 2 + 1]
                    obstacles.append([obs_x, obs_y])

            if not obstacles:
                obstacles = [[1,1]]
            
            up_scaled_matrix, up_inflated_matrix, up_black_ranges = Update2(
                original_scaled_matrix, original_black_ranges, obstacles
            )
            with lock:
                count_coll, count_failed, success, path_taken, best_route, k_value, stop_flag, count_routes = (
                    agent_navigation_step(
                        agent_id=agent_id,
                        routes=routes,
                        best_route=best_route,
                        end_point=end_point,
                        current_position=current_position,
                        up_black_ranges=up_black_ranges,
                        k_value=k_value,
                        agent_collision_grid=agent_collision_grid,
                        up_inflated_matrix=up_inflated_matrix,
                        iteration_collision_grid=np.zeros_like(up_inflated_matrix),
                        path_taken=path_taken,
                        up_scaled_matrix=up_scaled_matrix
                    )
                )
                
                # Update statistics
                shared_stats["COUNT_ALL_COLL"] += count_coll
                shared_stats["COUNT_ALL_FAILED"] += count_failed
                shared_stats["COUNT_ALL_ROUTES"] += count_routes
            
            if path_taken:
                current_position = (path_taken[-1])
                position_array[agent_id * 2] = current_position[0]
                position_array[agent_id * 2 + 1] = current_position[1]
            with lock:
                if success:
                    task_counts[agent_id] += 1 
                    completed_tasks += 1
                    shared_stats["TOTAL_TASKS_COMPLETED"] += 1
                    shared_env["total_collision_grid"] += agent_collision_grid
                
            if count_failed:
                failed_tasks += 1
            with lock:
                # Update agent state in shared memory
                agent_state.update({
                    "current_position": current_position,
                    "tasks": tasks,
                    "completed_tasks": completed_tasks,
                    "collisions": collisions,
                    "failed_tasks": failed_tasks,
                })

    print(f"%%% Agent {agent_id}: mission:{completed_tasks} , failed:{failed_tasks}, in total: mission: {shared_stats['TOTAL_TASKS_COMPLETED']}  failed: {shared_stats['COUNT_ALL_FAILED']} ")        

