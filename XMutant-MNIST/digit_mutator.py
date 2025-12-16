import random
import xml.etree.ElementTree as ET

import mutation_manager
import rasterization_tools
import vectorization_tools
from config import (  # MUTATION_RECORD,; MUTATION_TYPE,; MUTEXTENT,
    CONTROL_POINT,
    MUTOPPROB,
    SQUARE_SIZE,
)
from utils import closest_2d_point  # , get_distance

NAMESPACE = "{http://www.w3.org/2000/svg}"


class DigitMutator:

    def __init__(self, digit):
        """
        digit: Class Individual
        """
        self.digit = digit
        self.control_point = None

        root = ET.fromstring(self.digit.xml_desc)
        self.svg_path = root.find(NAMESPACE + "path").get("d")
        self.mutant_vector = self.svg_path
        self.mutation_points = None
        self.directions = None

    """
    Output one control point to be mutated based on selected method 
    """

    def control_points_select(self, selection_base):
        # mutate(self.digit.purified, self.digit.xml_desc, self.digit.attention, mutation, MUTEXTENT)

        # decide endpoints or intermediate points to mutate
        rand_mutation_probability = random.uniform(0, 1)
        if rand_mutation_probability >= MUTOPPROB:
            self.mutation_points = "end"
        else:
            self.mutation_points = "mid"

        # gather all candidates to be mutated
        control_points = mutation_manager.end_or_middle_points(self.svg_path, self.mutation_points)

        if len(control_points) <= 0:
            self.mutation_points = "end" if self.mutation_points == "mid" else "mid"
            control_points = mutation_manager.end_or_middle_points(
                self.svg_path, self.mutation_points
            )

        if len(control_points) <= 0:
            raise Exception("There is no mutation candidate")

        # select one point
        if selection_base == "random":
            # choose control point with equal weighting
            self.control_point = random.choice(control_points)
        elif selection_base == "square-window":
            # choose control point based on the XAI guided weights
            if self.digit.attention is None:
                raise Exception("calculate attention map first")

            weight_list = mutation_manager.get_attention_region_prob(
                self.digit.attention, control_points, SQUARE_SIZE
            )
            # print(f"weight_list: {weight_list}")
            if weight_list is not None:
                self.control_point = random.choices(
                    population=control_points, weights=weight_list, k=1
                )[0]

        elif selection_base == "clustering":
            # choose control point based on the XAI guided weights
            if self.digit.attention is None:
                raise Exception("calculate attention map first")

            (
                clustered_points,
                weight_list,
                directions,
                previous_percentage,
                cluster_rel_intensity,
            ) = mutation_manager.cluster_attention_map(
                self.digit.attention,
                control_points,
                weight_threshold=0.0001,
                previous_mask=self.digit.cluster_mask,
            )
            # print(f"weight_list: {weight_list}")
            if weight_list is not None:
                index = random.choices(
                    population=range(len(control_points)), weights=weight_list, k=1
                )[0]
                self.control_point = control_points[index]
                self.directions = directions[index]

                # ------------------for attention reduction-------------------------
                if self.digit.mutation_point[0] is not None:
                    # print(control_points)
                    # print(self.digit.mutation_point)
                    pre_index = control_points.index(
                        closest_2d_point(self.digit.mutation_point, control_points)
                    )
                    # print(pre_index)
                    # self.digit.avg_intensity_after = cluster_sum_intensity[pre_index]
                    self.digit.rel_intensity_after = previous_percentage

                # self.digit.avg_intensity_before = cluster_sum_intensity[index]
                self.digit.rel_intensity_before = cluster_rel_intensity[index]
                self.digit.cluster_mask = clustered_points == index

        self.digit.mutation_point = self.control_point

    def mutate(self):

        # in case you forget to choose control points
        if self.control_point is None:
            self.control_points_select(CONTROL_POINT)

        original_coordinates_str = None
        mutated_coordinates_str = None

        flag_dnf = True
        # Note: the endpoints should be within the range of [0,28],
        # while the intermediate points are not necessary.
        if self.mutation_points == "end":
            # backward_centroid will reach the boundary easily, which will lead to infinite loop...
            original_coordinates_str, mutated_coordinates_str, mutated_coordinate = (
                mutation_manager.mutate_one_point(self.control_point, direction=self.directions)
            )

            if type(mutated_coordinate) is str and 0 <= float(mutated_coordinate) <= 28:
                # valid mutation
                flag_dnf = False
            elif type(mutated_coordinate) is tuple:
                if (
                    0 <= float(mutated_coordinate[0]) <= 28
                    and 0 <= float(mutated_coordinate[1]) <= 28
                ):
                    flag_dnf = False

        elif self.mutation_points == "mid":
            original_coordinates_str, mutated_coordinates_str, mutated_coordinate = (
                mutation_manager.mutate_one_point(self.control_point, direction=self.directions)
            )
            flag_dnf = False
        if (
            original_coordinates_str is None
            or original_coordinates_str not in self.svg_path
            or flag_dnf
        ):
            raise Exception("Oops, cannot find the position of mutated point")

        mutant_vector = self.svg_path.replace(
            str(original_coordinates_str), str(mutated_coordinates_str)
        )

        self.digit.reset()
        self.digit.xml_desc = vectorization_tools.create_svg_xml(mutant_vector)
        self.digit.purified = rasterization_tools.rasterize_in_memory(self.digit.xml_desc)
        # NOTE: after mutation, AM should be recalculated.

        # print(f"Mutation: {original_coordinates_str} -> {mutated_coordinates_str}")
