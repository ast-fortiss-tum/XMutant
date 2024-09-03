import random
from typing import Tuple, List

import numpy as np
from shapely.geometry import Point
import warnings
from random import randint, uniform
from typing import List, Tuple

from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import Point

from driving.udacity_road import UdacityRoad
from utils.randomness import set_random_seed
from utils.visualization import RoadTestVisualizer
from config import ROAD_WIDTH, NUM_SAMPLED_POINTS, NUM_CONTROL_NODES, MAX_ANGLE, SEG_LENGTH, DISPLACEMENT
from driving.catmull_rom import catmull_rom
from global_log import GlobalLog
from driving.bbox import RoadBoundingBox

import math
import numpy as np

from driving.road import Road
from driving.road_polygon import RoadPolygon
from test_generators.test_generator import TestGenerator

warnings.simplefilter("ignore", ShapelyDeprecationWarning)

from driving.bbox import RoadBoundingBox
from driving.catmull_rom import catmull_rom
from driving.road_polygon import RoadPolygon
from utils.visualization import RoadTestVisualizer
from driving.udacity_road import UdacityRoad
from config import ROAD_WIDTH, NUM_SAMPLED_POINTS, NUM_CONTROL_NODES, MAX_ANGLE, SEG_LENGTH


logger = GlobalLog("road_utils")

def get_closest_control_point(point: Tuple[float, float], cp: List[Tuple[float, float]]) -> int:
    nodes = np.asarray(cp)
    dist_2 = np.sum((nodes - point) ** 2, axis=1)
    return int(np.argmin(dist_2))

def get_closest_previous_control_point(point: Tuple[float, float], cp: List[Tuple[float, float]]) -> int:
    nodes = np.asarray(cp)
    dist_2 = np.sum((nodes - point) ** 2, axis=1)
    nodes = np.argsort(dist_2)[0:2]
    return int(min(nodes))


def mutate_road(index: int,
                cp: List[Tuple[float, float, float, float]],
                method=None,
                direction=List[Tuple[float, float]]) \
        -> List[Tuple[float, float, float, float]]:
    """
    method:
        None - random
        "orthogonal_curve" - orthogonal to the curve and increase the curvature
        "orthogonal_random" - orthogonal to the curve and randomly select left or right
        "orthogonal_L" - orthogonal to the curve and move to left
        "orthogonal_R" - orthogonal to the curve and move to right
        "directional" - given direction
    """
    #print(cp)
    temp = list(cp[index])
    print(f"Idx {index} before mutation: {temp}")
    if method is None:
        # random mutation
        print("MUTATION ROAD: random angle")
        angle = np.random.uniform(low=0, high=2 * np.pi)
        dx = DISPLACEMENT * np.cos(angle)
        dy = DISPLACEMENT * np.sin(angle)
        temp[0] += dx
        temp[1] += dy

    elif method == "directional":
        assert np.linalg.norm(direction) > 0, f"Direction cannot be zero vector {direction}"
        normalized_direction = np.array(direction)
        normalized_direction = normalized_direction / np.linalg.norm(normalized_direction)
        dx = DISPLACEMENT * normalized_direction[0]
        dy = DISPLACEMENT * normalized_direction[1]
        temp[0] += dx
        temp[1] += dy
        print(f"Idx {index} after mutation: {temp}")
    else:
        # orthogonal directions
        if index == 0:
            point1 = np.array(cp[1])
            point2 = np.array(cp[2])
        elif index == len(cp)-1:
            point1 = np.array(cp[index-2])
            point2 = np.array(cp[index-1])
        else:
            point1 = np.array(cp[index-1])
            point2 = np.array(cp[index+1])
        orthogonal_left, orthogonal_right, orthogonal_increase = orthogonal_direction(temp, point1, point2)
        if method == "orthogonal_curve":
            print("MUTATION ROAD: orthogonal to the curve and increase the curvature")
            direction = orthogonal_increase

        elif method == "orthogonal_random":
            print("MUTATION ROAD: orthogonal to the curve and random side")
            direction = random.choice([orthogonal_left, orthogonal_right])

        elif method == "orthogonal_L":
            print("MUTATION ROAD: orthogonal to the curve and LEFT side")
            direction = orthogonal_left
        elif method == "orthogonal_R":
            print("MUTATION ROAD: orthogonal to the curve and RIGHT side")
            direction = orthogonal_right
        else:
            print("unknown type")
            return None
        print(f"mutate direction is {direction[0:2]}")
        return mutate_road(index, cp, method="directional", direction=direction[0:2])

    cp[index] = tuple(temp)
    #print(cp)
    return cp


def orthogonal_direction(mutation_point, point1, point2):
    # Gram-Schmidt process
    # print(temp, point1, point2)
    vec1 = mutation_point - point1
    vec2 = point2 - point1  # direction of car
    proj = np.dot(vec1, vec2) / np.dot(vec2, vec2) * vec2
    orthogonal_increase = vec1 - proj
    orthogonal_increase = orthogonal_increase[0:2]

    orthogonal_left = np.array([-vec2[1], vec2[0]])
    orthogonal_right = np.array([-vec2[1], vec2[0]])

    if np.linalg.norm(orthogonal_increase) == 0:
        orthogonal_increase = orthogonal_left

    return orthogonal_left, orthogonal_right, orthogonal_increase

"""def mutate_road(index: int, cp: List[Tuple[float, float, float, float]], dir=None) -> List[Tuple[float, float, float, float]]:
    temp = list(cp[index])
    print(f"Idx {index} before mutation: {temp}")
    if np.random.rand() < 0.5:
        temp[0] = temp[0] + DISPLACEMENT
    else:
        temp[1] = temp[1] + DISPLACEMENT

    print(f"Idx {index} after mutation: {temp}")

    cp[index] = tuple(temp)

    return cp"""

def is_valid(control_nodes, sample_nodes):
    return (RoadPolygon.from_nodes(sample_nodes).is_valid() and
            RoadBoundingBox(bbox_size=(0, 0, 250, 250)).contains(RoadPolygon.from_nodes(control_nodes[1:-1])))


if __name__ == '__main__':
    control_points_road = [(125.0, 0.0, -28.0, 8.0), (125.0, 25.0, -28.0, 8.0), (125.0, 50.0, -28.0, 8.0),
                           (128.91086162600578, 74.69220851487844, -28.0, 8.0),
                           (104.15415990746652, 78.17153603888008, -28.0, 8.0),
                           (79.29111252325968, 80.78474762057142, -28.0, 8.0),
                           (60.712491886324834, 97.51301277954289, -28.0, 8.0),
                           (36.2588018679797, 92.3152205090989, -28.0, 8.0),
                           (11.805111849634557, 87.11742823865492, -28.0, 8.0)]

    point_start = (60, 88, -28.0, 8.0)

    road_test_visualizer = RoadTestVisualizer(map_size=250)

    print(control_points_road)

    condition = True
    control_nodes = control_points_road
    while condition:
        control_nodes = control_nodes[1:]
        sample_nodes = catmull_rom(control_nodes, NUM_SAMPLED_POINTS)
        if is_valid(control_nodes, sample_nodes):
            condition = False

    road_points = [Point(node[0], node[1]) for node in sample_nodes]
    control_points = [Point(node[0], node[1], node[2]) for node in control_nodes]

    road_test_visualizer.visualize_road_test(
        road=UdacityRoad(road_width=ROAD_WIDTH,
                         road_points=road_points,
                         control_points=control_points),
        folder_path='../',
        filename='road',
        plot_control_points=False
    )

    closest = get_closest_control_point(tuple([point_start[0], point_start[1]]),
                                        [tuple([node[0], node[1]]) for node in control_nodes])
    print(closest)

    mutated = mutate_road(closest, control_points_road)

    print(mutated)

    condition = True
    control_nodes = mutated
    while condition:
        control_nodes = control_nodes[1:]
        sample_nodes = catmull_rom(control_nodes, NUM_SAMPLED_POINTS)
        if is_valid(control_nodes, sample_nodes):
            condition = False

    road_points = [Point(node[0], node[1]) for node in sample_nodes]
    control_points = [Point(node[0], node[1], node[2]) for node in control_nodes]

    road_test_visualizer.visualize_road_test(
        road=UdacityRoad(road_width=ROAD_WIDTH, road_points=road_points, control_points=control_points),
        folder_path='../',
        filename='road',
        plot_control_points=False
    )
