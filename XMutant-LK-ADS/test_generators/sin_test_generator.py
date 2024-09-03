# This code is used in the paper
# "Model-based exploration of the frontier of behaviours for deep learning system testing"
# by V. Riccio and P. Tonella
# https://doi.org/10.1145/3368089.3409730
import time
from typing import List

import numpy as np
from shapely.geometry import Point

from utils.visualization import RoadTestVisualizer
from config import ROAD_WIDTH, MAX_ANGLE
from driving.udacity_road import UdacityRoad
from global_log import GlobalLog
from test_generators.test_generator import TestGenerator
from utils.randomness import set_random_seed


class SinTestGenerator(TestGenerator):

    def __init__(self, max_angle=MAX_ANGLE):
        super().__init__(map_size=0)
        self.logg = GlobalLog('SinTestGenerator')
        self.max_angle = max_angle

    def generate(self):
        road_points: List[Point] = [Point(x, np.sin(x / 10) * 10) for x in np.arange(0., 900., 2.)]
        return UdacityRoad(
            road_points=road_points,
            control_points=[Point(p.x, p.y, 0.0) for p in road_points],
            road_width=ROAD_WIDTH,
        )

    def set_max_angle(self, max_angle: int) -> None:
        assert max_angle > 0, 'Max angle must be > 0. Found: {}'.format(max_angle)
        self.max_angle = max_angle


if __name__ == "__main__":
    map_size = 250

    set_random_seed(seed=0)

    roadgen = SinTestGenerator()
    start_time = time.perf_counter()
    road = roadgen.generate()
    concrete_representation = road.get_concrete_representation()
    print(time.perf_counter() - start_time)

    road_test_visualizer = RoadTestVisualizer(map_size=map_size)
    road_test_visualizer.visualize_road_test(road=road, folder_path='../', filename='road_2')
