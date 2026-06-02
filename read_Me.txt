בס"ד.

📂 Code Structure & Execution Instructions

This project includes three main simulation entry points (Main_* files), each designed for different execution modes. It also includes two types of agent logic and three function modules, depending on the algorithm and purpose.

🚀 Main Execution Files (Main_*.py)

Main_time.py

Runs the simulation based on a time limit (in minutes).
Suitable when you want the agents to operate for a fixed duration.


Main_iter.py

Runs the simulation based on a fixed number of completed tasks (iterations).
Ideal for consistent comparisons across simulations.


🤖 Agent Code

Each type of simulation uses a dedicated agent logic file:

For Main_time.py and Main_iter.py → use agent_navigation.py

These files define the behavior and task handling of agents during the simulation.


🧠 Function Modules

The system supports two algorithm types (Hybrid and RRT), and you must import the correct one in both the Main_*.py and agent_navigation*.py files to ensure consistency.

Algorithm Type		Function File to Use	Notes
Hybrid Algorithm	Functions_code.py	Works with Main_time or Main_iter
RRT Algorithm		Functions_code_RRT.py	Works with Main_time or Main_iter


⚠️ Consistency Warning
The function file import must be consistent across both the Main file and the agent logic file.
If you import, for example, Functions_code_RRT in Main_time.py, make sure agent_navigation.py imports the same one.
Mixing function files may lead to unexpected behavior or errors.