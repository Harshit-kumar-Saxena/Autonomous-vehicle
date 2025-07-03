import math

# Calculate Haversine distance (in meters) between two GPS coordinates
def haversine(coord1, coord2):
    R = 6371000  # Earth radius in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c

# Calculate bearing (direction) from one GPS point to another
def calculate_bearing(lat1, lon1, lat2, lon2):
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(delta_lambda)

    bearing = math.atan2(y, x)
    return (math.degrees(bearing) + 360) % 360

# Difference between target bearing and current heading
def heading_difference(bearing, heading):
    diff = bearing - heading
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return diff
