import json

def save_waypoints(path, nodes, filename="waypoints.json"):
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
