import random
import xml.etree.ElementTree as ET
import re
from random import randint, uniform
from config import MUTLOWERBOUND, MUTUPPERBOUND, MUTOFPROB,  MUTEXTENT, XMUTANT_CONFIG # MUTATION_TYPE,
import numpy as np

MUTATION_TYPE = XMUTANT_CONFIG["direction"]
NAMESPACE = '{http://www.w3.org/2000/svg}'


def apply_displacement_to_mutant(value, extent):
    displ = uniform(MUTLOWERBOUND, MUTUPPERBOUND) * extent
    if random.uniform(0, 1) >= MUTOFPROB:
        result = float(value) + displ
    else:
        result = float(value) - displ
    return repr(result)


# def apply_mutoperator1(svg_path, extent):
#
#     while(True):
#         # find all the vertexes
#         pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
#         segments = pattern.findall(svg_path)
#         svg_iter = re.finditer(pattern, svg_path)
#         # chose a random vertex
#         num_matches = len(segments) * 2
#
#         random_coordinate_index = randint(0, num_matches - 1)
#         # print(random_coordinate_index)
#
#         vertex = next(value for index, value in enumerate(svg_iter) if int(index == int(random_coordinate_index / 2)))
#         group_index = (random_coordinate_index % 2) + 1
#
#         value = apply_displacement_to_mutant(vertex.group(group_index), extent)
#
#         if 0 <= float(value) <= 28:
#             break
#
#     path = svg_path[:vertex.start(group_index)] + value + svg_path[vertex.end(group_index):]
#     return path


# def apply_mutoperator2(svg_path, extent):
#     # find all the vertexes
#     pattern = re.compile('C\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s')
#     segments = pattern.findall(svg_path)
#
#     # chose a random control point
#     num_matches = len(segments) * 4
#     path = svg_path
#     if num_matches > 0:
#         random_coordinate_index = randint(0, num_matches - 1)
#         svg_iter = re.finditer(pattern, svg_path)
#         control_point = next(value for index, value in enumerate(svg_iter) if int(index == int(random_coordinate_index/4)))
#         group_index = (random_coordinate_index % 4) + 1
#         value = apply_displacement_to_mutant(control_point.group(group_index), extent)
#         path = svg_path[:control_point.start(group_index)] + value + svg_path[control_point.end(group_index):]
#     else:
#         print("ERROR")
#         print(svg_path)
#     return path


# def mutate(svg_desc, operator_name, mutation_extent):
#     root = ET.fromstring(svg_desc)
#     svg_path = root.find(NAMESPACE + 'path').get('d')
#     mutant_vector = svg_path
#     if operator_name == 1:
#         mutant_vector = apply_mutoperator1(svg_path, mutation_extent)
#     elif operator_name == 2:
#         mutant_vector = apply_mutoperator2(svg_path, mutation_extent)
#     return mutant_vector

#
def generate(svg_desc, operator_name):
    root = ET.fromstring(svg_desc)
    svg_path = root.find(NAMESPACE + 'path').get('d')
    if operator_name == 1:
        vector1, vector2 = apply_operator1(svg_path)
    elif operator_name == 2:
        vector1, vector2 = apply_operator2(svg_path)
    return vector1, vector2


# def apply_displacement(value):
#     displ = uniform(MUTLOWERBOUND, MUTUPPERBOUND)
#     result = float(value) + displ
#     difference = float(value) - displ
#     return repr(result), repr(difference)


# def apply_operator1(svg_path):
#     while(True):
#         # find all the vertexes
#         pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
#         segments = pattern.findall(svg_path)
#         svg_iter = re.finditer(pattern, svg_path)
#         # chose a random vertex
#         num_matches = len(segments) * 2
#
#         random_coordinate_index = randint(0, num_matches - 1)
#
#         vertex = next(value for index, value in enumerate(svg_iter) if int(index == int(random_coordinate_index / 2)))
#         group_index = (random_coordinate_index % 2) + 1
#
#         value1, value2 = apply_displacement(vertex.group(group_index))
#
#         if 0 <= float(value1) <= 28 and 0 <= float(value2) <= 28:
#             break
#
#     path1 = svg_path[:vertex.start(group_index)] + value1 + svg_path[vertex.end(group_index):]
#     path2 = svg_path[:vertex.start(group_index)] + value2 + svg_path[vertex.end(group_index):]
#     return path1, path2


# def apply_operator2(svg_path):
#     # find all the vertexes
#     pattern = re.compile('C\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s')
#     segments = pattern.findall(svg_path)
#
#     # chose a random control point
#     num_matches = len(segments) * 4
#     path1 = svg_path
#     path2 = svg_path
#     if num_matches > 0:
#         random_coordinate_index = randint(0, num_matches - 1)
#         svg_iter = re.finditer(pattern, svg_path)
#         control_point = next(value for index, value in enumerate(svg_iter) if int(index == int(random_coordinate_index/4)))
#         group_index = (random_coordinate_index % 4) + 1
#         value1, value2 = apply_displacement(control_point.group(group_index))
#         path1 = svg_path[:control_point.start(group_index)] + value1 + svg_path[control_point.end(group_index):]
#         path2 = svg_path[:control_point.start(group_index)] + value2 + svg_path[control_point.end(group_index):]
#     else:
#         print("ERROR")
#         print(svg_path)
#     return path1, path2


def end_or_middle_points(svg_path, mode):
    """
    Input:

    Output:
    """
    control_points = []
    if mode == "end":
        pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
        segments = pattern.findall(svg_path)
        #svg_iter = re.finditer(pattern, svg_path)
        control_points = [(float(i[0]), float(i[1])) for i in segments]
        #print(control_points)
    elif mode == "mid":
        pattern = re.compile('C\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s')
        segments = pattern.findall(svg_path)

        for i in segments:
            control_points.append((float(i[0]), float(i[1])))
            control_points.append((float(i[2]), float(i[3])))
        #print(control_points)
    return control_points


def cluster_attention_map(map,
                          control_points,
                          weight_threshold=0.1,
                          # previous_mask=None
                          ):
    control_points = np.array(control_points)
    centroids = np.zeros(control_points.shape)
    centroids[:, 0] = control_points[:, 1]
    centroids[:, 1] = control_points[:, 0]

    num_rows = 28
    num_cols = 28
    points = np.array([(i, j) for i in range(num_rows) for j in range(num_cols)])
    weights = map.flatten()

    # if previous_mask is not None:
    #     previous_percentage = weights[previous_mask].sum() / (weights.sum())
    # else:
    #     previous_percentage = None

    # use euclidean distance to cluster heatmap
    distances = np.sqrt(((points - centroids[:, np.newaxis]) ** 2).sum(axis=2))
    distances_to_centroid = np.min(distances, axis=0)
    clustered_points = np.argmin(distances, axis=0)

    clustered_points[weights < weight_threshold] = -1


    # points = np.array([(i,j) for i in range(num_rows) for j in range(num_cols)])
    # clustered_points = clustered_points.reshape(num_rows, num_rows)

    # compute the weights
    centroids_weight = np.zeros(centroids.shape[0])
    new_centroids = np.zeros(centroids.shape)
    # cluster_avg_intensity = np.zeros(centroids.shape[0]) # for reduction of attention
    # cluster_sum_intensity = np.zeros(centroids.shape[0]) # for reduction of attention

    for i in range(centroids.shape[0]):
        points_in_cluster = points[clustered_points == i]
        weights_in_cluster = weights[clustered_points == i]
        # cluster_avg_intensity[i] = np.mean(weights_in_cluster) # for reduction of attention
        # cluster_sum_intensity[i] = np.sum(weights_in_cluster) # for reduction of attention

        # TODO np.sum(weights_in_cluster) can be zero sometimes
        if np.sum(weights_in_cluster) > 0:
            new_centroids[i] = np.sum(points_in_cluster.T * weights_in_cluster, axis=1) / np.sum(weights_in_cluster)

            distances_in_cluster = distances_to_centroid[clustered_points == i]
            distances_in_cluster[distances_in_cluster < 1] = 1

            weights_by_distances = np.multiply(weights_in_cluster, 1. / np.power(distances_in_cluster, 1))

            centroids_weight[i] = np.sum(weights_by_distances)
            # print(f"centroids_weight {i} {centroids_weight[i]}")

        else:
            new_centroids[i] = np.array([14, 14])
            centroids_weight[i] = 0
    # cluster_rel_intensity = cluster_sum_intensity/cluster_sum_intensity.sum()
    # softmax
    weight_list = np.exp(centroids_weight / np.sum((centroids_weight), axis=0) *30)
    if MUTATION_TYPE == "toward_centroid" or "backward_centroid":
        directions = new_centroids - control_points
        for i, dir in enumerate(directions):
            dir_norm = np.linalg.norm(dir)
            # print(dir_norm)
            directions[i] = np.array([0, 0]) if dir_norm == 0 or np.isnan(dir_norm) else dir / dir_norm
    else:
        directions = None
    return clustered_points, weight_list, directions # , previous_percentage, cluster_rel_intensity


def mutate_one_point(point, direction=None, mutation_direction = MUTATION_TYPE):
    """
    point: (float, float)
    """
    original_coordinates_str = str(point[0]) + "," + str(point[1])
    if mutation_direction == "random":
        # select X or Y to be mutated
        original_coordinate = random.choice(point)
        # + or - a random number uniformly distributed in [0.01, 0.6] to the coordinate.
        mutated_coordinate = apply_displacement_to_mutant(original_coordinate, MUTEXTENT)
        mutated_coordinates_str = original_coordinates_str.replace(str(original_coordinate),
                                                                   str(mutated_coordinate))
        return original_coordinates_str, mutated_coordinates_str, mutated_coordinate
    elif mutation_direction == "random_cycle":
        length = uniform(MUTLOWERBOUND, MUTUPPERBOUND) * MUTEXTENT
        angle = uniform(0, 2 * np.pi)
        dx = length * np.cos(angle)
        dy = length * np.sin(angle)
        x_mutant = point[0] + dx
        y_mutant = point[1] + dy
        mutated_coordinates_str = str(x_mutant) + "," + str(y_mutant)
        return original_coordinates_str, mutated_coordinates_str, (x_mutant, y_mutant)
    elif mutation_direction == "toward_centroid" or "backward_centroid" or "centroid_based":
        #assert direction[0] is float
        #assert direction[1] is float
        assert direction is not None
        #print(direction)
        length = uniform(MUTLOWERBOUND, MUTUPPERBOUND) * MUTEXTENT
        if mutation_direction == "toward_centroid":
            dx = length * direction[0]
            dy = length * direction[1]
        elif mutation_direction == "backward_centroid":
            dx = -length * direction[0]
            dy = -length * direction[1]
        else:
            if random.uniform(0, 1) >= 0.5:
                dx = length * direction[0]
                dy = length * direction[1]
            else:
                dx = -length * direction[0]
                dy = -length * direction[1]
        x_mutant = point[0] + dx
        y_mutant = point[1] + dy
        mutated_coordinates_str = str(x_mutant) + "," + str(y_mutant)

        return original_coordinates_str, mutated_coordinates_str, (x_mutant, y_mutant)
