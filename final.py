import tkinter as tk
from tkinter import messagebox
import pygame
import random
import math
from collections import deque 
from queue import PriorityQueue
import networkx as nx

# Constants for maze elements
COIN_VALUES = [5, 10]
POTHOLE_VALUES = [3, 6]
GOAL_VALUES = [1, 2, 3]
BARRIER_VALUE = "BARRIER"

# Probabilities for maze elements
COIN_PROBABILITY = 0.15
POTHOLE_PROBABILITY = 0.1
GOAL_PROBABILITY = 0.01
BARRIER_PROBABILITY = 0.1

class Maze:
    def __init__(self, rows, cols, cell_size, flag_coordinates, agent_coordinates):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.grid = [[None for _ in range(cols)] for _ in range(rows)]
        self.coin_positions = []
        self.pothole_positions = []
        self.flag_coordinates = flag_coordinates
        self.coin_img = pygame.image.load("coin.png").convert_alpha()  # Load coin image
        self.pothole_img = pygame.image.load("pothole.png").convert_alpha()  # Load pothole image
        self.goal_img = pygame.image.load("flag.png").convert_alpha()  # Load goal image
        self.agent_img = pygame.image.load("agent.png").convert_alpha()  # Load agent image
        self.resize_images()
        self.generate_maze()

        self.avoid_coordinates = []  # List to store coordinates of potholes encountered
        self.dead_end_memory = set()

        self.agents = [Agent(x, y, algo) for x, y, algo in agent_coordinates]
    
    def neighbors(self, cell):
        x, y = cell
        neighbors = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.cols and 0 <= new_y < self.rows:
                neighbors.append((new_x, new_y))
        return neighbors

    def resize_images(self):
        self.coin_img = pygame.transform.scale(self.coin_img, (self.cell_size, self.cell_size))
        self.pothole_img = pygame.transform.scale(self.pothole_img, (self.cell_size, self.cell_size))
        self.goal_img = pygame.transform.scale(self.goal_img, (self.cell_size, self.cell_size))
        self.agent_img = pygame.transform.scale(self.agent_img, (self.cell_size, self.cell_size))

    def generate_maze(self):
        # Place flags first
        for idx, (x, y) in enumerate(self.flag_coordinates):
            self.grid[y][x] = random.choice(GOAL_VALUES)
        
        for i in range(self.rows):
            for j in range(self.cols):
                # Skip if already occupied by a flag
                if self.grid[i][j] in GOAL_VALUES:
                    continue
                
                rand = random.random()
                if rand < COIN_PROBABILITY:
                    coin_value = random.choice(COIN_VALUES)
                    self.grid[i][j] = coin_value
                    if coin_value in COIN_VALUES:
                        self.coin_positions.append((i, j))
                elif rand < COIN_PROBABILITY + POTHOLE_PROBABILITY:
                    if self.grid[i][j] not in GOAL_VALUES:
                        pothole_value = random.choice(POTHOLE_VALUES)
                        self.grid[i][j] = pothole_value
                        if pothole_value in POTHOLE_VALUES:
                            self.pothole_positions.append((i, j))
                elif rand < COIN_PROBABILITY + POTHOLE_PROBABILITY + BARRIER_PROBABILITY:
                    self.grid[i][j] = BARRIER_VALUE
                else:
                    self.grid[i][j] = "Empty"

    def display_maze(self, screen):
        for i in range(self.rows):
            for j in range(self.cols):
                value = self.grid[i][j]
                cell_rect = pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                if value in COIN_VALUES:
                    screen.blit(self.coin_img, cell_rect)
                elif value in POTHOLE_VALUES:
                    screen.blit(self.pothole_img, cell_rect)
                elif value in GOAL_VALUES:
                    screen.blit(self.goal_img, cell_rect)
                elif value == BARRIER_VALUE:
                    pass
        
        # Draw flags
        for flag_x, flag_y in self.flag_coordinates:
            screen.blit(self.goal_img, (flag_x * self.cell_size, flag_y * self.cell_size))

        # Draw agents
        for agent in self.agents:
            screen.blit(self.agent_img, (agent.x * self.cell_size, agent.y * self.cell_size))

        # Draw vertical grid lines
        for i in range(self.cols):
            pygame.draw.line(screen, (0, 0, 0), (i * self.cell_size, 0), (i * self.cell_size, screen.get_height()))

        # Draw horizontal grid lines
        for i in range(self.rows):
            pygame.draw.line(screen, (0, 0, 0), (0, i * self.cell_size), (screen.get_width(), i * self.cell_size))

    def check_agent_position(self, agent):
        x, y = agent.x, agent.y
        cell_content = self.grid[y][x]

        if cell_content in COIN_VALUES:
            agent.reward += cell_content
            self.grid[y][x] = "Empty"  # Remove coin from maze
            self.coin_positions.remove((y, x))  # Remove coin position from list

        elif cell_content in POTHOLE_VALUES:
            agent.reward -= cell_content
            self.avoid_coordinates.append((x, y))

        elif cell_content in GOAL_VALUES:
            agent.reached_goal = True



class Agent:
    def __init__(self, x, y, algorithm):
        self.x = x
        self.y = y
        self.algorithm = algorithm
        self.dfs_i = True
        self.reached_goal = False
        self.reward = 0  # Initialize reward for the agent
        self.visited = set()  # Set to store visited cells
        self.frontier = deque()  # Queue to store frontier cells
        self.dfs_list = []
        
    def mark_dead_end(self, maze, position):
        maze.dead_end_memory.add(position)

    def is_dead_end(self, maze, position):
        return position in maze.dead_end_memory

    def move_randomly(self, maze):
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.choice(directions)
        for dx, dy in directions:
            new_x, new_y = self.x + dx, self.y + dy
            if (new_x, new_y) in maze.avoid_coordinates or self.is_dead_end(maze, (new_x, new_y)):
                continue
            if 0 <= new_x < maze.cols and 0 <= new_y < maze.rows:
                if maze.grid[new_y][new_x] != BARRIER_VALUE:
                    self.x, self.y = new_x, new_y
                    return
        self.mark_dead_end(maze, (self.x, self.y))
                
    
    def bfs(self, maze):
        self.frontier.append((self.x, self.y))
        self.visited.add((self.x, self.y))
        
        while self.frontier:
            current = self.frontier.popleft()
            if current in maze.flag_coordinates:
                self.reached_goal = True
                return

            neighbors = [(current[0] + dx, current[1] + dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]]
            for nx, ny in neighbors:
                if 0 <= nx < maze.cols and 0 <= ny < maze.rows and (nx, ny) not in self.visited \
                        and maze.grid[ny][nx] != BARRIER_VALUE and (nx, ny) not in maze.avoid_coordinates \
                        and not self.is_dead_end(maze, (nx, ny)):
                    self.frontier.append((nx, ny))
                    self.visited.add((nx, ny))
            if not self.frontier:
                self.mark_dead_end(maze, current)
    
    def create_graph(self,N, M):
        G = nx.Graph()
        for i in range(N):
            for j in range(M):
                node = (i, j)
                G.add_node(node)
                if i > 0:
                    G.add_edge(node, (i-1, j))
                if j > 0:
                    G.add_edge(node, (i, j-1))
        return G

    def dfs_traversal(self, graph, start_node, maze):
        stack = [start_node]
        visited = set()
        path = []

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                path.append(node)

                if node in maze.flag_coordinates:
                    return path

                neighbors = list(graph.neighbors(node))
                neighbors = [n for n in neighbors if not self.is_dead_end(maze, n)]
                if not neighbors:
                    self.mark_dead_end(maze, node)
                stack.extend(neighbors[::-1])  # Reverse to maintain DFS order

        # If no path found, mark all nodes as dead ends
        for node in path:
            self.mark_dead_end(maze, node)
        return []

    def mov_dfs(self, maze):
        if self.dfs_list:
            x, y = self.dfs_list.pop(0)
            if (x, y) not in maze.avoid_coordinates and not self.is_dead_end(maze, (x, y)):
                self.x, self.y = x, y
                if (self.x, self.y) in maze.flag_coordinates:
                    self.reached_goal = True
            else:
                self.mark_dead_end(maze, (x, y))

    
    def heuristic(self, p1, p2,heuristic_type):
        if heuristic_type == 1:
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        elif heuristic_type == 2:
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        elif heuristic_type == 3:
            return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
        else:
            raise ValueError("Invalid heuristic type.")

    def a_star(self, maze, heuristic_type):
        start = (self.x, self.y)
        goal = maze.flag_coordinates[0]  # Assuming one flag for simplicity

        frontier = PriorityQueue()
        frontier.put((0, start))
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0

        while not frontier.empty():
            _, current = frontier.get()

            if current == goal:
                break

            neighbors = maze.neighbors(current)
            neighbors = [n for n in neighbors if not self.is_dead_end(maze, n) and n not in maze.avoid_coordinates and maze.grid[n[1]][n[0]] != BARRIER_VALUE]
            if not neighbors:
                self.mark_dead_end(maze, current)
                continue

            for next in neighbors:
                new_cost = cost_so_far[current] + 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next, heuristic_type)
                    frontier.put((priority, next))
                    came_from[next] = current

        # Reconstruct path only if the goal is reached
        if current == goal:
            path = []
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()

            if len(path) > 1:
                next_step = path[1]
                self.x, self.y = next_step


            
    def hill_climbing(self, maze, heuristic_type):
        current = (self.x, self.y)
        goal = maze.flag_coordinates[0]
        current_cost = self.heuristic(current, goal, heuristic_type)
        
        neighbors = maze.neighbors(current)
        next_step = None
        next_cost = current_cost

        for neighbor in neighbors:
            if maze.grid[neighbor[1]][neighbor[0]] != BARRIER_VALUE and not self.is_dead_end(maze, neighbor):
                neighbor_cost = self.heuristic(neighbor, goal, heuristic_type)
                if neighbor_cost < next_cost:
                    next_step = neighbor
                    next_cost = neighbor_cost

        if next_step is not None:
            self.x, self.y = next_step
            current = next_step

        if current == goal:
            self.reached_goal = True


    def move(self, maze, algorithm_type, heuristic_type):
        if self.reached_goal:
            return

        if algorithm_type == 'R':
            self.move_randomly(maze)
        elif algorithm_type == 'B':
            self.bfs(maze)
        elif algorithm_type == 'D':
            if self.dfs_i:
                graph = self.create_graph(maze.rows, maze.cols)
                start_node = (self.x, self.y)
                self.dfs_list = self.dfs_traversal(graph, start_node, maze)
                self.dfs_i = False
            self.mov_dfs(maze)
        elif algorithm_type == 'A':
            self.a_star(maze, heuristic_type)
        elif algorithm_type == 'H':
            self.hill_climbing(maze, heuristic_type)

        maze.check_agent_position(self)



class FlagAgentGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Flag Agent GUI")
        self.num_flags = 0
        self.num_agents = 0
        self.flags = []
        self.agents = []

        self.flag_agent_frame = tk.Frame(self.master)
        self.flag_agent_frame.pack()

        self.flag_label = tk.Label(self.flag_agent_frame, text="Enter the number of flags:")
        self.flag_label.grid(row=0, column=0, sticky="e")
        self.flag_entry = tk.Entry(self.flag_agent_frame)
        self.flag_entry.grid(row=0, column=1)

        self.agent_label = tk.Label(self.flag_agent_frame, text="Enter the number of agents:")
        self.agent_label.grid(row=1, column=0, sticky="e")
        self.agent_entry = tk.Entry(self.flag_agent_frame)
        self.agent_entry.grid(row=1, column=1)

        self.flag_agent_button = tk.Button(self.flag_agent_frame, text="Submit", command=self.get_flags_agents)
        self.flag_agent_button.grid(row=2, columnspan=2)

    def get_flags_agents(self):
        self.num_flags = int(self.flag_entry.get())
        self.num_agents = int(self.agent_entry.get())

        self.flag_coordinates_frame = tk.Frame(self.master)
        self.flag_coordinates_frame.pack()

        for i in range(self.num_flags):
            x_label = tk.Label(self.flag_coordinates_frame, text=f"Enter x-coordinate for flag {i+1}:")
            x_label.grid(row=i, column=0, sticky="e")
            x_entry = tk.Entry(self.flag_coordinates_frame)
            x_entry.grid(row=i, column=1)
            self.flags.append(x_entry)

            y_label = tk.Label(self.flag_coordinates_frame, text=f"Enter y-coordinate for flag {i+1}:")
            y_label.grid(row=i, column=2, sticky="e")
            y_entry = tk.Entry(self.flag_coordinates_frame)
            y_entry.grid(row=i, column=3)
            self.flags.append(y_entry)

        self.agent_frame = tk.Frame(self.master)
        self.agent_frame.pack()

        for i in range(self.num_agents):
            x_label = tk.Label(self.agent_frame, text=f"Enter starting x-coordinate for agent {i+1}:")
            x_label.grid(row=i, column=0, sticky="e")
            x_entry = tk.Entry(self.agent_frame)
            x_entry.grid(row=i, column=1)
            self.agents.append(x_entry)

            y_label = tk.Label(self.agent_frame, text=f"Enter starting y-coordinate for agent {i+1}:")
            y_label.grid(row=i, column=2, sticky="e")
            y_entry = tk.Entry(self.agent_frame)
            y_entry.grid(row=i, column=3)
            self.agents.append(y_entry)

            algo_label = tk.Label(self.agent_frame, text=f"Select the algorithm for agent {i+1}:")
            algo_label.grid(row=i, column=4, sticky="e")
            algo_var = tk.StringVar(self.master)
            algo_var.set("R")
            algo_optionmenu = tk.OptionMenu(self.agent_frame, algo_var, "R", "B", "D", "A", "H")
            algo_optionmenu.grid(row=i, column=5)

            self.agents.append(algo_var)

        self.submit_button = tk.Button(self.master, text="Submit", command=self.submit_data)
        self.submit_button.pack()

    def submit_data(self):
        flag_coordinates = []
        for i in range(0, len(self.flags), 2):
            try:
                x = int(self.flags[i].get())
                y = int(self.flags[i + 1].get())
                flag_coordinates.append((x, y))
            except ValueError:
                messagebox.showerror("Error", "Please enter valid coordinates for all flags.")
                return

        agent_data = []
        for i in range(0, len(self.agents), 3):
            try:
                x = int(self.agents[i].get())
                y = int(self.agents[i + 1].get())
                algo = self.agents[i + 2].get()
                agent_data.append((x, y, algo))
            except ValueError:
                messagebox.showerror("Error", "Please enter valid coordinates for all agents.")
                return
            except IndexError:
                messagebox.showerror("Error", "Please select an algorithm for all agents.")
                return

        print("Flags:")
        for flag in flag_coordinates:
            print(flag)
        print("\nAgents:")
        for agent in agent_data:
            print(agent)

        self.start_game(flag_coordinates, agent_data)

    def start_game(self, flag_coordinates, agent_coordinates):
        pygame.init()

        rows, cols = 10, 10
        cell_size = 50
        screen_width, screen_height = cols * cell_size, rows * cell_size
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Maze Game")

        maze = Maze(rows, cols, cell_size, flag_coordinates, agent_coordinates)

        font = pygame.font.Font(None, 20)

        running = True
        clock = pygame.time.Clock()
        heuristic_type = 1
        a_flag = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill((255, 255, 255))
            maze.display_maze(screen)

            for i, agent in enumerate(maze.agents):
                if not agent.reached_goal:
                    if agent.algorithm == 'A' or agent.algorithm == 'H':
                        if not a_flag:
                            a_flag = True
                            heuristic_type = int(input("Choose heuristic distance (1: Manhattan, 2: Euclidean, 3: Chebyshev): "))
                            while heuristic_type not in [1, 2, 3]:
                                heuristic_type = int(input("Invalid input. Please choose 1, 2, or 3: "))
                    agent.move(maze, agent.algorithm, heuristic_type)
                    print(f"Agent {i+1} - Final Cost: {agent.reward}")

            pygame.display.flip()
            clock.tick(2)

        pygame.quit()
        for i, agent in enumerate(maze.agents):
            print(f"Agent {i+1} - Goal Reached with Cost: {agent.reward}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FlagAgentGUI(root)
    root.mainloop()