import scipy
import shapely
import numpy as np
import skimage as ski
import warnings
from sklearn.cluster import KMeans 

# 1) shapeToImg
# 2) sortedCentroidToLine
# order_points
# 3) get_angle 
# extendLine 
# 4) addPoints 
# extendLine 
# 5) centerline 

def shapeToImg(
    polygon: shapely.Polygon, 
    micron_to_pixel: float = 1,
    only_position: bool = True,
) -> tuple | np.ndarray: 
    """Create a binary image representing the polygon.

    Parameters
    ----------
    polygon
        Polygon to transform
    micron_to_pixel
        Size of one micron in pixel
    only_position
        If return only the position of the shape or also the binary image of the shape 
        
    Return
    ----------
        img: binary image
        posImg: value x,y of all pixel True in the binary image
    """
    boundary_pixels = np.rint(
        (shapely.get_coordinates(polygon.boundary)) / micron_to_pixel)
    
    x = int(boundary_pixels[:, 0].max()) + 1
    y = int(boundary_pixels[:, 1].max()) + 1

    # Create a binary image representing the polygon
    img = np.zeros((y, x), dtype=bool)
    row, col = ski.draw.polygon(*boundary_pixels.T)
    img[col, row] = 1
    # row = y
    # col = x
    yy, xx = np.where(img)
    posImg = np.column_stack((xx, yy))
    if only_position: 
        return posImg
    else:
        return img, posImg


def order_points(
    dist_matrix, 
    start=0
) -> list:
    """Function to find the TSP path


    Args:
        dist_matrix (_type_): _description_
        start (int, optional): _description_. Defaults to 0.

    Returns:
        list: _description_
    """
    n_points = len(dist_matrix) # dist_matrix.shape[0]
    ordered_points = [start]
    visited = [False] * n_points
    visited[start] = True
    
    for _ in range(1, n_points):
        min_distance = float('inf')
        next_point = None
        
        for j in range(n_points):
            if not visited[j] and dist_matrix[start, j] < min_distance:
                min_distance = dist_matrix[start, j]
                next_point = j
        
        if next_point is None:  # No further points can be connected
            break
        
        ordered_points.append(next_point)
        visited[next_point] = True
        start = next_point
    
    if len(ordered_points) < n_points:
        return None
    else:
        return ordered_points


def sortedCentroidToLine(
    polygon: shapely.Polygon, 
    centroids,
    length_max: int = 5,
) -> shapely.LineString:
    """Re order the points of all the centroids to have a 
    linestring without crossing paths

    Args:
        polygon (shapely.Polygon): _description_
        centroids (_type_): _description_
        length_max (int, optional): _description_. Defaults to 5.

    Returns:
        shapely.LineString: _description_
    """
    n = len(centroids)
    min_dist = float('inf')
    # min_order = []
    min_ordered_line = None

    dist_matrix = scipy.spatial.distance_matrix(centroids, centroids)

    for i in range(n):
        for j in range(i+1, n):
            line = shapely.LineString([centroids[i], centroids[j]])
            # if polygon.boundary.intersects(line):
            #     dist_matrix[i, j] = float('inf')
            #      dist_matrix[j, i] = float('inf')
            if polygon.boundary.intersects(line):
                if line.difference(polygon).length > length_max:
                    # warnings.warn(f'Lline unauthorized, line intersects polygon and the length is > {length_max}') 
                    dist_matrix[i, j] = float('inf')
                    dist_matrix[j, i] = float('inf')
                else :
                    warnings.warn(f"Line authorized, line intersects polygon but the length is < {length_max}") 

    for i in range(n):
        ordered_points = order_points(dist_matrix, start=i)
        if ordered_points:
            ordered_array = centroids[ordered_points]
            ordered_line = shapely.LineString(ordered_array)
            if ordered_line.is_simple:
                path_length = ordered_line.length
                
                if path_length < min_dist:
                    min_dist = path_length 
                    # min_order = ordered_array
                    min_ordered_line = ordered_line
                    
    return min_ordered_line


def addPoints(
    polygon: shapely.Polygon, 
    line: shapely.LineString, 
    dict_position: dict = {'Start': [0,1], 
                           'End': [-1,-2]}, 
    distance: int= 5000
) -> shapely.LineString:
    """

    Parameters:
        polygon (shapely.Polygon): _description_
        line (shapely.LineString): _description_
        dict_position (_type_, optional): _description_. Defaults to {'Start': [0,1], 'End': [-1,-2]}.
        distance (int, optional): _description_. Defaults to 5000.

    Returns:
        shapely.LineString: _description_
    """
    points = {}
    order_centers = shapely.get_coordinates(line)

    for loc, pos in dict_position.items():
        print(f'Add point at the {loc} position : {pos}')
        extendedLine = extendLine(order_centers[pos[0], :], 
                                  order_centers[pos[1], :], distance)
        touch_bound = shapely.get_coordinates(polygon.boundary.intersection(extendedLine))

        if len(touch_bound) > 1 :
            min_dist = float('inf')
            print(f"The {loc} touches 2 points : keep the closest")
            for point in touch_bound:
                dist_pts = shapely.distance(shapely.Point(point), 
                                            shapely.Point(order_centers[pos[0]]))
                if dist_pts < min_dist:
                    # print(f"Smaller distance for the {point = } : {dist_pts}")
                    min_dist = dist_pts
                    points[loc] = point
        else:
            points[loc] = touch_bound

    lineFinal = shapely.LineString(
        np.vstack([points['Start'], order_centers, points['End']]))
    return lineFinal


def extendLine(
    point1: shapely.Point, 
    point2: shapely.Point, 
    distance: int = 5000,
) -> shapely.LineString:
    """    Extend a line segment formed by two points by a given distance.

    Parameters
    ----------
    point1, point2: Tuple representing the coordinates of the two points
    distance: Distance by which to expand the line

    Return
    ----------
    Line representing the coordinates of the expanded line segment
    """
    p1 = np.array(point1)
    p2 = np.array(point2)

    # Compute the direction vector of the line and normalize the direction vector
    direction = p2 - p1
    direction = direction / np.linalg.norm(direction)

    # Compute the new points by adding/subtracting the direction vector
    new_point1 = p1 - direction * distance
    new_point2 = p2 + direction * distance

    line = shapely.LineString([
        new_point1,
        point1,
        point2,
        new_point2])
    return line


def get_angle(
    p1: shapely.Point, 
    p2: shapely.Point, 
    p3: shapely.Point, 
    degree: bool = True,
) -> float: 
    """Calculate angle between 2 points

    Args:
        p1 (shapely.Point): _description_
        p2 (shapely.Point): _description_
        p3 (shapely.Point): _description_
        degree (bool, optional): _description_. Defaults to True.

    Returns:
        float: _description_
    """
    vec1 = p1 - p2
    vec2 = p3 - p2
    
    cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(cosine_angle)

    if degree:
        return np.degrees(angle)
    else:
        return angle
    
         
def centerline(
    polygon: shapely.Polygon,
    n_clusters: int = 3,
    distance: int = 5000,
    length_max: int = 5,
    random_state: int = 130,
    img_val = None,
    # seuil: float = 0.75,
    threshold: int = 120,
    # max_clusters = 100,
) -> shapely.LineString:
    """Compute the centerline

    Args:
        polygon (shapely.Polygon): _description_
        n_clusters (int, optional): _description_. Defaults to 3.
        distance (int, optional): _description_. Defaults to 5000.
        length_max (int, optional): _description_. Defaults to 5.
        random_state (int, optional): _description_. Defaults to 130.
        img_val (_type_, optional): _description_. Defaults to None.
        threshold (int, optional): _description_. Defaults to 120.

    Returns:
        shapely.LineString: _description_
    """
 
    if img_val is None :
        img_val = shapeToImg(polygon=polygon)
    print("===========================================")
    print(f"Start research of the best k...")
    # threshold=None
    search = True
    lineK_Order = None
    
    while search:
        print(f"n_clusters = {n_clusters}...")

        model = KMeans(n_clusters= n_clusters, random_state = random_state).fit(img_val)
        lineK_1_Order = sortedCentroidToLine(polygon, model.cluster_centers_, length_max=length_max)
    
        if lineK_1_Order:
            if lineK_Order:
                coordinates = shapely.get_coordinates(lineK_1_Order)
                angles=[]
                
                if not lineK_1_Order.is_simple:
                    warnings.warn("Warning Message: the line is not simple") 

                for i in range(1, len(coordinates) - 1):
                    angle = get_angle(p1=coordinates[i-1], 
                                    p2=coordinates[i], 
                                    p3=coordinates[i+1], 
                                    degree = True)
                    angles.append(angle)
                    
                # val = np.argwhere(angles < np.quantile(angles,0.20))
                # for i in range(len(val)-1):
                #     if val[i] + 1 == val[i+1]:
                #         print(f'Best n_clusters found = {n_clusters-1}')
                #         n_clusters -= 1 
                #         search = False
                #         break   
                
                val = np.argwhere(np.array(angles) < threshold)
                for i in range(len(val)-1):
                    if (val[i] + 1 == val[i+1]) | (val[i] + 2 == val[i+1]) :
                        print(f'Best n_clusters found = {n_clusters-1}')
                        n_clusters -= 1 
                        search = False
                        break   
                if not search:
                    break
                
            n_clusters += 1
            lineK_Order = lineK_1_Order
        
        else:
            print("No line...")
            n_clusters += 1
           
    lineFinal = addPoints(polygon = polygon, 
                            line = lineK_Order, distance = distance)
    return lineFinal
