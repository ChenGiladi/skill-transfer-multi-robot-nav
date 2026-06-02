import sys, os
sys.path.append("C:\\Users") # add path (harmless on non-Windows)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

try:
    from torchmetrics import ConfusionMatrix
except ImportError:
    ConfusionMatrix = None  # unused in the simulation path (training-only leftover)
import seaborn as sns # For plotting the confusion matrix 
import matplotlib.pyplot as plt # For plotting the confusion matrix

from scipy.ndimage import binary_dilation
import math

from sklearn.ensemble import RandomForestRegressor
import joblib
import copy

from scipy.spatial.distance import euclidean

import matplotlib.colors as mcolors

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from matplotlib.ticker import MultipleLocator
import time


# Run RRT

show_animation=False
class RRT:
    """
    Class for RRT planning
    """

    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    
    def __init__(self,
                start,
                goal,
                black_ranges, # 24.10 @@@ add
                matrix,
                expand_dis=1.0,
                path_resolution=0.5,
                goal_sample_rate=5,
                max_iter=int(os.environ.get("RRT_MAX_ITER", "8000")),
                robot_radius=0.0,
                stop_dist=0.7):  # Added stop_dist for flexibility
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.matrix = np.array(matrix)
        self.rows, self.cols = self.matrix.shape
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = []
        self.robot_radius = robot_radius
        self.stop_dist = stop_dist
        self.black_ranges = black_ranges  # Store up_black_ranges

    def planning(self, animation=True):
        """
        RRT path planning
        """
        self.node_list = [self.start]
        best_node_index = 0  # Track the index of the best (closest) node
        best_dist = self.calc_dist_to_goal(self.start.x, self.start.y)  # Initial distance to goal

        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node):
                # Round the x and y coordinates of the node before adding it  @@@ we add this 8/10
                new_node.x = int(round(new_node.x))
                new_node.y = int(round(new_node.y))
                
                # Append the rounded node to the node list
                self.node_list.append(new_node)





                # Calculate distance to goal for the new node
                dist_to_goal = self.calc_dist_to_goal(new_node.x, new_node.y)
                # Update the closest node if necessary
                if dist_to_goal < best_dist:
                    best_node_index = len(self.node_list) - 1
                    best_dist = dist_to_goal

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

            # Check if we reached within the stopping distance
            if best_dist <= self.stop_dist:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.check_collision(final_node):
                    return self.generate_final_course(len(self.node_list) - 1)

        # If we exit the loop, return the path to the closest node
        pass  # (silenced) RRT reached max iterations
        return [] # self.generate_final_course(best_node_index)

    # (Rest of the code remains unchanged)
    
    
    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        if extend_length > d:
            extend_length = d
        n_expand = math.floor(extend_length / self.path_resolution)
        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)
        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y
        new_node.parent = from_node
        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(0, self.cols-1),
                random.uniform(0, self.rows-1))
        else:
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def check_collision(self, node):
        """
        Check if the node is in a collision-free position
        Now, 0 is safe for movement, and 1 indicates an obstacle.
        """
        if node is None:
            return False
        for (x, y) in zip(node.path_x, node.path_y):
            x_index, y_index = int(round(x)), int(round(y))
            if not (0 <= x_index < self.cols and 0 <= y_index < self.rows):
                return False  # Out of bounds
            
            # Check for collision using is_point_inside_black_ranges
            if is_point_inside_black_ranges(x_index, y_index, self.black_ranges): 
                return False  # Collision with an obstacle
        return True  # Safe



    def draw_graph(self, rnd=None):
        plt.clf()
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")  # Plot random points (if needed)
        
        # Draw all the nodes and paths
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")
        
        # Plot the start point in blue
        plt.plot(self.start.x, self.start.y, "ob", markersize=10)  # 'ob' -> blue circle
        plt.text(self.start.x, self.start.y, "Start", fontsize=12, verticalalignment='bottom', horizontalalignment='right', color='blue')

        # Plot the goal point in red
        plt.plot(self.end.x, self.end.y, "or", markersize=10)  # 'or' -> red circle
        plt.text(self.end.x, self.end.y, "Goal", fontsize=12, verticalalignment='bottom', horizontalalignment='right', color='red')

        # Plot the matrix/grid
        plt.imshow(1 - self.matrix, cmap='gray', origin='lower')  # Reverse color mapping to show obstacles
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind


# Lightweight, opt-in instrumentation: counts online-RRT planner invocations per episode.
# Reset to 0 by the simulator before each episode and read afterwards; does not affect navigation.
RRT_CALL_COUNT = 0


def Apply_RRT(current_position, end_point, matrix, up_black_ranges):
    global RRT_CALL_COUNT
    RRT_CALL_COUNT += 1

    safe_route_found = True
    rrt = RRT(current_position, end_point, up_black_ranges, matrix=np.flipud(matrix))
    # Search for the path
    rrt_route = rrt.planning(animation=show_animation)
    best_route = rrt_route[::-1]

 
    k_values = 0
    if best_route is None:
        pass  # (silenced) RRT: no path
        safe_route_found = False
    seen = set()  # To track unique points
    filtered_rrt_route = []
    for point in best_route:
        if tuple(point) not in seen:  # Convert point to tuple (since list is unhashable)
            seen.add(tuple(point))     # Mark point as seen
            filtered_rrt_route.append(point)  # Add unique point to the final route
    best_route = filtered_rrt_route[1:-1]
    return best_route, k_values, safe_route_found


# ----------------------------------BC PART----------------------------------------------------------------------------

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_actions):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 64)
        self.output_layer = nn.Linear(64, num_actions)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return out

# Load the trained model
input_size = 29
num_actions = 2
model = NeuralNetwork(input_size, num_actions)
model1_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "trained_model_NN.pth")

model.load_state_dict(torch.load(model1_path, map_location="cpu"))

def extract_local_environment(maze, center_of_mass_position):

    if isinstance(center_of_mass_position[0], torch.Tensor):
        # Convert tensor elements to Python scalars
        x = center_of_mass_position[0].item()
        y = center_of_mass_position[1].item()
    else:
        # Elements are numbers, directly assign them
        x, y = center_of_mass_position
    # Round each element to the nearest integer
    x = round(x)
    y = round(y) 

    row= maze.shape[0] - y - 1
    col=x
    
    local_environment = maze[row-2:row+3, col-2:col+3]
    local_environment = local_environment.astype(int)
    return local_environment


def run_BC_agent_NN(start_COM_point, goal_point, up_scaled_matrix, up_black_ranges, model):
    
    COM_positions_arr = []
    maze = up_scaled_matrix
    done = False
    first_run = True
    num_steps = 0


    while not done:
        num_steps += 1
        observations = []

        if first_run:
            current_COM_position = start_COM_point
            first_run = False

        if current_COM_position[0] > 50 or current_COM_position[1] > 50 or current_COM_position[0] < 0 or current_COM_position[1] < 0:
            break
            
        if is_point_inside_black_ranges(current_COM_position[0], current_COM_position[1], up_black_ranges):
            break

        distance_to_goal = np.sqrt((current_COM_position[0] - goal_point[0])**2 + (current_COM_position[1] - goal_point[1])**2)    

        if distance_to_goal <=1.5: # 1.5:
            done = True

        # Create observations only if the position is valid and goal is not reached
        local_environment = extract_local_environment(maze, current_COM_position)
        observations.extend([current_COM_position[0], current_COM_position[1], goal_point[0], goal_point[1]])
        observations.extend(local_environment.flatten(order='F'))

        # Convert observations list to PyTorch tensor
        observations_tensor = torch.tensor(observations, dtype=torch.float32)
        observations_tensor2 = observations_tensor.unsqueeze(0)

        model.eval()
        # Generate predictions
        with torch.no_grad():
            # Send observations array to model in eval mode and receive actions
            outputs = model(observations_tensor2)
            actions =outputs 

        # Append current positions to lists
        COM_positions_arr.append(current_COM_position)

        # Update position
        current_COM_position = torch.tensor(current_COM_position, dtype=torch.float32)
        current_COM_position += actions.squeeze()
        current_COM_position = current_COM_position.tolist()

        # If the number of steps exceeds 100, terminate
        if num_steps > 100:
            break


    if done == 1:
        return 1, COM_positions_arr
    else:
        return 0, COM_positions_arr

def Apply_BC(model, start_point, goal_point, up_scaled_matrix, up_black_ranges):
    current_position = start_point

    agent_success, route= run_BC_agent_NN(current_position, goal_point, up_scaled_matrix, up_black_ranges, model) # NN model
    if agent_success == 1:
        return True, route[1:]   # Success
    else:
        return False, route[1:]  # Failure




def Update1(original_scaled_matrix, original_black_ranges, new_obs_list):
    
    black_ranges = copy.deepcopy(original_black_ranges)
    up_black_ranges = update_black_ranges1(black_ranges, new_obs_list)
    # Make a copy of the scaled matrix
    up_scaled_matrix = np.copy(original_scaled_matrix)

    # Inflate obstacles in the occupancy grid using binary_dilation
    structuring_element = np.ones((3, 3))
    up_inflated_matrix = binary_dilation(up_scaled_matrix, structure=structuring_element).astype(up_scaled_matrix.dtype)

    for new_obs in new_obs_list:
        # Convert (x, y) to matrix indices, handling non-integer values
        row_obs = int(up_scaled_matrix.shape[0] - round(new_obs[1]) - 1)
        col_obs = int(round(new_obs[0]))

        # Add the new obstacle to the scaled matrix
        up_scaled_matrix[row_obs, col_obs] = 1

        # Add the new obstacle after dilation
        up_inflated_matrix[row_obs, col_obs] = 1

    return up_scaled_matrix, up_inflated_matrix, up_black_ranges


def update_black_ranges1(black_ranges, obstacles):
    for x, y in obstacles:
        x = int(x)
        y = int(y)
        
        x_range = (x - 1, x + 2)
        
        y_lower = y - 1
        y_upper = y + 2 +1

        for y_val in range(y_lower, y_upper):
            if 0 <= y_val < len(black_ranges):
                range_added = False
                for i, (start, end) in enumerate(black_ranges[y_val]):
                    if start <= x_range[0] <= end or start <= x_range[1] <= end:
                        start = min(start, x_range[0])
                        end = max(end, x_range[1])
                        black_ranges[y_val][i] = (start, end)
                        range_added = True
                        break
                if not range_added:
                    black_ranges[y_val].append(x_range)
                    black_ranges[y_val] = sorted(black_ranges[y_val])
    
    return black_ranges

def Update2(original_scaled_matrix, original_black_ranges, new_obs_list):

    black_ranges = copy.deepcopy(original_black_ranges)
    up_black_ranges = update_black_ranges1(black_ranges, new_obs_list)

    up_scaled_matrix = np.copy(original_scaled_matrix)
    up_inflated_matrix = np.copy(original_scaled_matrix)

    for new_obs in new_obs_list:
        new_obs = [int(new_obs[0]), int(new_obs[1])]
  
        row_obs = int(up_scaled_matrix.shape[0] - new_obs[1] - 1)
        col_obs = int(new_obs[0])

        if 0 <= row_obs < up_scaled_matrix.shape[0] and 0 <= col_obs < up_scaled_matrix.shape[1]:
            up_scaled_matrix[row_obs, col_obs] = 1
            up_inflated_matrix[row_obs, col_obs] = 1

    structuring_element = np.ones((3, 3))
    up_inflated_matrix = binary_dilation(up_inflated_matrix, structure=structuring_element).astype(up_inflated_matrix.dtype)

    return up_scaled_matrix, up_inflated_matrix, up_black_ranges

def is_point_inside_black_ranges(x, y, up_black_ranges):

    def is_in_black_range(x, y_val):
        if y_val < 0 or y_val >= len(up_black_ranges):
            return False
        for x_range in up_black_ranges[y_val]:
            if x_range[0] < x < x_range[1]:
                return True
        return False

    if not isinstance(y, int):
        y_floor = math.floor(y)
        y_ceil = math.ceil(y)
        
        # Check both y_floor and y_ceil
        in_floor = is_in_black_range(x, y_floor)
        in_ceil = is_in_black_range(x, y_ceil)

        # Return True only if both conditions are satisfied
        return in_floor and in_ceil

    else:
        # For integer y, check if it's in the black ranges
        return is_in_black_range(x, y)
    
# Continue with route extraction (tis is for the original data)
def extract_routes(df):
    routes = []
    current_route = []
    
    for i, row in df.iterrows():
        # Check if the row is a start or end of a route
        if all(float(row[j]).is_integer() for j in range(2)):
            # If there's an ongoing route, finalize it
            if current_route:
                routes.append(current_route)
                current_route = []
        
        # Append row to the current route
        current_route.append(row.tolist())
    
    # Append the last route if it exists
    if current_route:
        routes.append(current_route)
    
    return routes


# For Alternative data
def extract_routes_newdata(df):
    routes = []
    current_route = []
    
    for i, row in df.iterrows():
        x, y = row[0], row[1]
        
        # Check if the row is the track separator (1,1)
        if x == 1 and y == 1:
            # If there's an ongoing route, finalize it
            if current_route:
                routes.append(current_route)
                current_route = []
        else:
            # Append row to the current route
            current_route.append([x, y])
    
    # Append the last route if it exists
    if current_route:
        routes.append(current_route)
    
    return routes

# Filter routes by proximity to the start and end points
def filter_by_proximity(routes, start_point, end_point, top_n):
    def route_distance(route):
        start_point_2d = route[0][:2]  # Extract 2D point if necessary
        end_point_2d = route[-1][:2]  # Extract 2D point if necessary
        start_dist = euclidean_distance(start_point_2d, start_point)
        end_dist = euclidean_distance(end_point_2d, end_point)
        return start_dist + end_dist
    
    sorted_routes = sorted(routes, key=route_distance)[:top_n]
    return sorted_routes


# Check if a point is within the bounding box
def is_point_in_bbox(point, bbox):
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]

# Extract sub-routes that pass through the bounding box
def extract_sub_routes(routes, bbox):
    def get_sub_route(route, bbox):
        sub_route = []
        for point in route:
            if is_point_in_bbox(point[:2], bbox):
                sub_route.append(point)
            elif sub_route:
                yield sub_route
                sub_route = []
        if sub_route:
            yield sub_route
    
    all_sub_routes = []
    for route in routes:
        for sub_route in get_sub_route(route, bbox):
            if len(sub_route) > 1:
                all_sub_routes.append(sub_route)
    return all_sub_routes

# Plot routes and bounding box
def plot_routes_and_bbox(proximity_filtered_routes, sub_routes, bbox, start_point, goal_point):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot bounding box
    rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                         linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)

    # Plot proximity filtered routes
    for i, route in enumerate(proximity_filtered_routes):
        route = np.array(route)[:, :2]  # Ensure only 2D points are used
        ax.plot(route[:, 0], route[:, 1], color='green', linestyle='-', marker='o', linewidth=2, label='Complete Route' if i == 0 else "")

    # Plot sub-routes
    for i, route in enumerate(sub_routes[:20]):  # Limit to 20 sub-routes
        route = np.array(route)[:, :2]  # Ensure only 2D points are used
        ax.plot(route[:, 0], route[:, 1], color='orange', linestyle='-', marker='o', linewidth=2, label='Sub-Route' if i == 0 else "")

    # Plot the start and goal points
    ax.plot(start_point[0], start_point[1], 'o', color='yellow', label='Start Point', markersize=12, markeredgewidth=4)
    ax.plot(goal_point[0], goal_point[1], 'rx', label='Goal Point', markersize=12, markeredgewidth=4)

    # Set grid with a jump of 1
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.grid(True, which='both', linestyle='--', linewidth=0.7)


    # Add legend manually to ensure all labels are included
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Routes and Sub-Routes with Bounding Box')
    ax.grid(True)
    plt.show()



def Get_best_routes(routes, current_position, end_point, up_black_ranges,
                    collision_grid=None, db_alpha=0.0):
    # collision_grid/db_alpha implement SELECTIVE COLLISION-HISTORY SHARING: a robot
    # with database access (collision_grid given, db_alpha>0) chooses, among the valid
    # candidate routes, the one that traverses the fewest historically collision-prone
    # cells; without access it keeps the original behaviour (first valid route).
    b=4
    scaling_factors = [2, 3, 6]

    def _congestion(rt):
        s = 0.0
        H, W = collision_grid.shape
        for pt in rt:
            cx = round(pt[0]); cy = round(pt[1])
            rr = H - cy - 1; cc = cx
            if 0 <= rr < H and 0 <= cc < W:
                s += collision_grid[rr, cc]
        return s

    for scale in scaling_factors:

        # Adjust bbox by the current scaling factor
        bbox2 = [min(current_position[0], end_point[0]) - scale*b,
                 min(current_position[1], end_point[1]) - scale*b,
                 max(current_position[0], end_point[0]) + scale*b,
                 max(current_position[1], end_point[1]) + scale*b]

        # Step 1: Filter by proximity
        sub_routes = extract_sub_routes(routes, bbox2)

        # Step 2: Filter combined routes by proximity again
        top_routes = filter_by_proximity(sub_routes, current_position, end_point, top_n=10)

        valid_routes = []
        current_distance_to_dest = euclidean_distance(current_position, end_point)
        for route in top_routes:
            route = Cut_route_closet_point(route, current_position)
            safe_route = True
            for step_ahead in range(1, 5):
                if step_ahead < len(route):
                    step_x, step_y = route[step_ahead][:2]
                    if is_point_inside_black_ranges(step_x, step_y, up_black_ranges):
                        safe_route = False
                        break
            if safe_route:
                new_end_point = min(route, key=lambda pt: euclidean_distance(pt, end_point))
                route = route[:route.index(new_end_point) + 1]
                dist_closest_to_dest = euclidean_distance(route[-1], end_point)
                if dist_closest_to_dest < current_distance_to_dest:  # Keep the route only if it brings the agent closer to the destination
                    if collision_grid is not None and db_alpha:
                        valid_routes.append(route)   # collect; choose least-congested below
                    else:
                        return [route, []]           # original behaviour: first valid route

        if collision_grid is not None and db_alpha and valid_routes:
            best = min(valid_routes, key=_congestion)   # avoid collision-prone cells
            return [best, []]
        return []


# Check if the point is at least 'min_distance' away from all points in the list
def is_valid_point(new_point, existing_points, min_distance):
    for point in existing_points:
        if euclidean_distance(new_point, point) < min_distance:
            return False
    return True

# Generate valid start and end points
def generate_start_and_end(matrix, up_black_ranges, num_agents, min_distance=4):

    def generate_valid_point():
        while True:
            x = random.randint(4, 46)
            y = random.randint(4, 46)
            if not is_point_inside_black_ranges(x, y, up_black_ranges):
                return [x, y]
    
    # Initialize lists to store start and end points for each agent
    start_points = []
    end_points = []
    
    for i in range(num_agents):
        # Generate valid start point
        while True:
            start_point = generate_valid_point()
            if is_valid_point(start_point, start_points, min_distance):  # Check distance from other start points
                start_points.append(start_point)
                break
        
        # Generate valid end point
        while True:
            end_point = generate_valid_point()
            if is_valid_point(end_point, end_points, min_distance) and euclidean_distance(start_point, end_point) >= min_distance:
                end_points.append(end_point)
                break
    
    return start_points, end_points

# Iterate over the collision grid and print positions of collisions
def print_collision_positions(collision_grid):
    height, width = collision_grid.shape
    for y in range(height):
        for x in range(width):
            if collision_grid[x, y] > 0:  # Check if there was a collision in this cell
                collision_grid[x, y]+=0 
                return collision_grid[x, y]

# Helper function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Adjust route navigation
def nav_help(routes, best_route, end_point, current_position, up_black_ranges, k, up_inflated_matrix, up_scaled_matrix, collision_grid=None, db_alpha=0.0):
    counter_all_nav =0
    if k >= len(best_route):
        top_routes = Get_best_routes(routes, current_position, end_point, up_black_ranges, collision_grid, db_alpha)
        if top_routes == []:
            return [], k, counter_all_nav
    
        best_route = top_routes[0]
        counter_all_nav = 1 # for the counter of all traj we take
        k = 0
        distance = euclidean_distance(current_position, best_route[0])
        if  distance>= 2:
            if distance<= 7:
                agent_success, bc_route = Apply_BC(model, current_position, best_route[0], up_scaled_matrix, up_black_ranges)
                if agent_success == True:
                    best_route = bc_route + best_route
                else:
                    best_route = []
                    return [], k, counter_all_nav
            else:
                rrt_route, _, _ = Apply_RRT(current_position, best_route[0], up_inflated_matrix, up_black_ranges) #apply online rrt
                best_route = rrt_route + best_route


    return best_route, k, counter_all_nav

# Cut route to closest point
def Cut_route_closet_point(best_route, current_position):
    min_index = min(range(len(best_route)), key=lambda i: euclidean_distance(best_route[i], current_position))
    best_route = best_route[min_index:]
    return best_route

# Replan if a collision is detected
def check_collision_and_replan(best_route, current_position, up_black_ranges, routes, end_point, k, up_inflated_matrix, up_scaled_matrix, collision_grid=None, db_alpha=0.0):

    original_route = best_route
    counter_All_coll=1  # for all  routes
    combined_routes = Get_best_routes(routes, current_position, end_point, up_black_ranges, collision_grid, db_alpha)

    if combined_routes == []:
        return original_route, k, False, counter_All_coll
    else:
        best_route = combined_routes[0]
        distance = euclidean_distance(current_position, best_route[0])
        if  distance>= 2:
            if distance<= 7:
                agent_success, bc_route = Apply_BC(model, current_position, best_route[0], up_scaled_matrix, up_black_ranges) #apply bc model

                if agent_success == True:
                    best_route = bc_route + best_route
                else:
                    best_route = original_route
                    return best_route, k, False, counter_All_coll
            else:
                rrt_route, _, _ = Apply_RRT(current_position, best_route[0], up_inflated_matrix, up_black_ranges) #apply online rrt 
                best_route = rrt_route + best_route


              
        return best_route, 0, True, counter_All_coll

# Plot the routes of agents
def plot_route(run_sim, path_taken, start_point, end_point, up_inflated_matrix, agent_id, original_route=None):
    colors = list(mcolors.TABLEAU_COLORS.values())
    cmap = mcolors.ListedColormap(['white', 'black'])
    height, width = up_inflated_matrix.shape
    flipped_matrix = np.flipud(up_inflated_matrix)
    plt.imshow(flipped_matrix, cmap=cmap, origin='lower', extent=[0, width, 0, height])

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.xticks(np.arange(0, width + 1, 1))
    plt.yticks(np.arange(0, height + 1, 1))

    path_taken = np.array(path_taken)
    
    # Plot the original route (if any) as dashed lines
    if original_route is not None:
        original_route = np.array(original_route)
        plt.plot(original_route[:, 0], original_route[:, 1], linestyle='--', color=colors[agent_id % len(colors)], label=f'Original route of Agent {agent_id+1}', linewidth=2)
    
    # Plot the path taken by the agent
    plt.plot(path_taken[:, 0], path_taken[:, 1], linestyle='-', marker='o', linewidth=2, 
             label=f'Path taken by Agent {agent_id+1}', color=colors[agent_id % len(colors)])
    
    # Add numbers next to each point on the path
    for i, (x, y) in enumerate(path_taken):
        plt.text(x, y, str(i+1), fontsize=9, color='black', ha='right')

    # Plot start and end points
    plt.plot(start_point[0], start_point[1], 'o', color=colors[agent_id % len(colors)], label='Start Point', markersize=10, markeredgewidth=2)
    plt.plot(end_point[0], end_point[1], 'x', color=colors[agent_id % len(colors)], label='End Point', markersize=10, markeredgewidth=2)

    plt.legend(loc='upper right')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Planned vs. Actual Routes of Agents in Collision-Aware Navigation (1)")


def save_all_collision_grids_single_file(collision_grid_store, filename='all_collision_grids.txt'):
    with open(filename, 'w') as f:
        for (num_agents, time_limit), collision_grid in collision_grid_store.items():
            # Write a header for each grid
            f.write(f'# Collision Grid for {num_agents} agents over {time_limit} minutes\n')
            np.savetxt(f, collision_grid, fmt='%d')
            f.write('\n')  # Add a newline between matrices for readability
    print(f"All collision grids saved to {filename}")

# plot heatMap for X agents in Y times
def plot_collision_heatmap(collision_grid, up_inflated_matrix, num_agents, time_limit, title="Collision Intensity with Radius Effect on Maze"):
    # Get the height and width from the environment matrix
    height, width = up_inflated_matrix.shape
    block_size = 1  # Block size for collision grid
    radius = 1.5    # Radius for the effect of each collision point
    num_points = 100  # Resolution of the heatmap grid

    # Generate a 2D grid for x and y bounds
    y, x = np.meshgrid(np.linspace(0, height, num_points), np.linspace(0, width, num_points))

    # Initialize the Z matrix for collision intensities
    z = np.zeros_like(x)

    # Loop through the collision grid and accumulate the intensity in the radius
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            # Check if there's any collision in the current block
            if collision_grid[i, j] > 0:
                x_cell = j
                y_cell = up_inflated_matrix.shape[0] - i - 1

                # Add a circular intensity effect around the collision point
                dist = np.sqrt((x - x_cell)**2 + (y - y_cell)**2)
                z += np.exp(-(dist / radius)**2) * collision_grid[i, j]

    # Normalize the Z matrix to make the color intensity smooth
    z_min, z_max = 0, np.max(z)

    # Plot using pcolormesh
    fig, ax = plt.subplots(figsize=(15, 15))

    # Base maze (static obstacles) - plot using imshow or pcolormesh
    cmap_base = mcolors.ListedColormap(['white', 'black'])
    plt.imshow(np.flipud(up_inflated_matrix), cmap=cmap_base, origin='lower', extent=[0, width, 0, height])

    # Plot the collision intensity heatmap using pcolormesh
    c = ax.pcolormesh(x, y, z, cmap='Reds', vmin=z_min, vmax=z_max, shading='auto', alpha=0.6)

    # Add a colorbar for the collision intensities
    cbar = fig.colorbar(c, ax=ax, label='Collision Intensity')
    cbar.ax.tick_params(labelsize=14)  # Increase the font size of colorbar ticks
    cbar.set_label('Collision Intensity', fontsize=14, fontweight='bold', labelpad=10)  # Bold label for colorbar

    # Set title and axis labels with bold font
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xlabel('X', fontsize=14, fontweight='bold')  # X-axis label
    ax.set_ylabel('Y', fontsize=14, fontweight='bold')  # Y-axis label
    
    # Save the plot as a high-quality PNG image
    filename = f'collision_heatmap_{num_agents}_agents_{time_limit}_minutes_maze2_data1.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=500)
    plt.show()


# Function to check if the distance between two points is greater than a threshold
def is_distance_greater_than(point1, point2, threshold=2):
    distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    return distance > threshold

# Function to filter routes that do not pass through black areas and meet distance criteria
def filter_routes_through_black_ranges_and_distance(routes, up_black_ranges, distance_threshold=2):
    filtered_routes = []
    
    for route in routes:
        # Check if any point in the route is in a black area
        if not any(is_point_inside_black_ranges(point[0], point[1], up_black_ranges) for point in route):
            # Check if the distance between any two consecutive points is greater than the threshold
            if all(not is_distance_greater_than(route[i], route[i + 1], distance_threshold) for i in range(len(route) - 1)):
                filtered_routes.append(route)
    
    return filtered_routes



# Function for a single agent's navigation logic
def agent_navigation_step(agent_id, routes, best_route, end_point, current_position, up_black_ranges, k_value, agent_collision_grid,
                           up_inflated_matrix, iteration_collision_grid, path_taken, up_scaled_matrix, collision_grid=None, db_alpha=0.0,
                           static_black_ranges=None, coll_type_counter=None):
    # static_black_ranges / coll_type_counter are opt-in instrumentation only: when both are
    # supplied, each counted collision is additionally classified as a static-obstacle vs a
    # robot-robot (dynamic) collision in coll_type_counter=[obstacle, robot]. Defaults (None)
    # leave behavior and the returned collision count bit-identical.
    count_coll = 0
    count_failed = 0
    count_routes = 0
    continue_agent = True
    success = False
    stop_flag = 0
 
    

    def calculate_path_distance(route):
        return sum(np.linalg.norm(np.array(route[i+1]) - np.array(route[i])) for i in range(len(route) - 1))

    # Only call check_collision_and_replan based on a probabilistic approach
    def should_replan():
        # Calculate the distance covered so far
        distance_covered = calculate_path_distance(path_taken)
        # Calculate the remaining distance in the current best route
        remaining_distance = calculate_path_distance(best_route[k_value:])
        # Calculate the percentage of path covered
        total_distance = distance_covered + remaining_distance
        if total_distance == 0:  # Avoid division by zero if no distance is available
            return False
        percentage_completed = distance_covered / total_distance
        # Draw a random number and decide based on the calculated percentage
        rand = random.random() 
        return rand > percentage_completed


    if euclidean_distance(current_position, end_point) >= 1.5 and continue_agent:
        best_route, k_value, counter_all_nav = nav_help(routes, best_route, end_point, current_position, up_black_ranges, k_value, up_inflated_matrix, up_scaled_matrix, collision_grid, db_alpha)
        count_routes+=counter_all_nav

        if best_route == []:
            count_failed += 1
            success = True
            stop_flag = 1
            return  count_coll, count_failed, success, path_taken, best_route, k_value, stop_flag,  count_routes
        
        
        if k_value + 3 < len(best_route):
            three_steps_ahead = best_route[k_value + 2][:2]
            if is_point_inside_black_ranges(three_steps_ahead[0], three_steps_ahead[1], up_black_ranges):
                count_coll += 1
                if coll_type_counter is not None:
                    if static_black_ranges is not None and is_point_inside_black_ranges(
                            three_steps_ahead[0], three_steps_ahead[1], static_black_ranges):
                        coll_type_counter[0] += 1   # static-obstacle collision
                    else:
                        coll_type_counter[1] += 1   # robot-robot (dynamic) collision
                row, col = up_inflated_matrix.shape[0] - round(three_steps_ahead[1]) - 1, round(three_steps_ahead[0])

                # Check if the cell has already been marked as a collision in this iteration
                if iteration_collision_grid[row, col] == 0:
                    agent_collision_grid[row, col] += 1
                    iteration_collision_grid[row, col] = 1  # Mark it as updated in this iteration
                    num_of_coll = agent_collision_grid[row, col]
                else:
                    num_of_coll = agent_collision_grid[row, col]  # Read the current value without increment
                if num_of_coll > 8:
                    count_failed += 1
                    success = True
                    stop_flag = 1
                    return  count_coll, count_failed, success, path_taken, best_route, k_value, stop_flag,  count_routes
                
                # Replanning due to potential collision
                if should_replan():
                    best_route, k_value, safe_route_found, counter_All_coll = check_collision_and_replan(
                        best_route, current_position, up_black_ranges, routes, end_point, k_value, up_inflated_matrix, up_scaled_matrix, collision_grid, db_alpha)
                # Update the agent's position and path taken
                    current_position = np.array(best_route[k_value][:2])
                    path_taken.append(current_position)
                    if safe_route_found:
                        k_value += 1
                        count_routes+=1
                  
                else:
                    current_position = np.array(best_route[k_value][:2])
                    path_taken.append(current_position)
            
                            
            else:
                # Update the agent's position and path taken
                current_position = np.array(best_route[k_value][:2])
                path_taken.append(current_position)
                k_value += 1
        else:
            current_position = np.array(best_route[k_value][:2])
            path_taken.append(current_position)
            k_value += 1
    else:
        success = True  # Agent has reached its destination

    return  count_coll, count_failed, success, path_taken, best_route, k_value, stop_flag,  count_routes

