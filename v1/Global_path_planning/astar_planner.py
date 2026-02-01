import heapq
import json
from haversin import haversine 
from global_path_utils import save_waypoints


with open("updated_graph.json", "r") as f:
    graph = json.load(f)

nodes = graph["nodes"]


def reconstruct_path(came_from, start, goal):
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def astar(nodes, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}

    g_score = {node: float("inf") for node in nodes}
    g_score[start] = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            save_waypoints(path, nodes)
            return path


        # if current == goal:
        #     path = []
        #     while current in came_from:
        #         path.append(current)
        #         current = came_from[current]
        #     path.append(start)
        #     path = path[::-1]

        #     # Save both name and coordinates
        #     waypoints = [
        #         {"name": node, "coord": nodes[node]["coord"]} for node in path
        #     ]
        #     save_waypoints(waypoints,nodes)

        #     return path
          

        for neighbor, cost in nodes[current]["neighbors"].items():
            tentative_g = g_score[current] + cost
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + haversine(nodes[neighbor]["coord"], nodes[goal]["coord"])
                heapq.heappush(open_set, (f, neighbor))

    return None

def get_user_input():
    start_node = input("Enter the start node: ")
    goal_node = input("Enter the goal node: ")
    return start_node, goal_node

# Main Program
if __name__ == "__main__":
    with open("updated_graph.json", "r") as f:
        graph = json.load(f)

    nodes = graph['nodes']

    start_node, goal_node = get_user_input()
    path = astar(nodes, start_node, goal_node)

    if path:
        print("Shortest path:", path)
    else:
        print("No path found.")
