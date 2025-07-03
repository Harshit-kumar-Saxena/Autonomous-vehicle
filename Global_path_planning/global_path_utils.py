import json

def save_waypoints(path, nodes, filename="waypoints.json"):
    """
    Save the list of waypoints with names and coordinates into a JSON file.

    Parameters:
    - path: List of node names from A* output.
    - nodes: Dictionary of all node data with coordinates.
    - filename: File name to save the waypoints. Default: 'waypoints.json'
    """
    waypoints = []

    for node in path:
        if node in nodes:
            waypoints.append({
                "name": node,
                "coord": nodes[node]["coord"]
            })
        else:
            print(f"[Warning] Node {node} not found in graph.")

    with open(filename, "w") as f:
        json.dump(waypoints, f, indent=4)

    print(f"[Info] Waypoints saved to {filename}")
