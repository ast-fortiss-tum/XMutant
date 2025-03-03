import random
import mutation_manager
import rasterization_tools
import vectorization_tools
from mnist_member import MnistMember
from config import MUTOPPROB, XMUTANT_CONFIG # CONTROL_POINT
from utils import get_distance, closest_2d_point

import xml.etree.ElementTree as ET
NAMESPACE = '{http://www.w3.org/2000/svg}'

class DigitMutator:

    def __init__(self, digit):
        self.digit = digit
        # self.seed = digit.seed

        root = ET.fromstring(self.digit.xml_desc)
        self.svg_path = root.find(NAMESPACE + 'path').get('d')
        self.directions = None
        self.control_point = None
        self.mutation_points_mode = None

    def control_points_select(self, selection_base):
        # mutate(self.digit.purified, self.digit.xml_desc, self.digit.attention, mutation, MUTEXTENT)

        # decide endpoints or intermediate points to mutate
        rand_mutation_probability = random.uniform(0, 1)
        if rand_mutation_probability >= MUTOPPROB:
            self.mutation_points_mode = "end"
        else:
            self.mutation_points_mode = "mid"

        # gather all candidates to be mutated
        control_points = mutation_manager.end_or_middle_points(self.svg_path, self.mutation_points_mode)

        if len(control_points) <= 0:
            self.mutation_points_mode = "end" if self.mutation_points_mode == "mid" else "mid"
            control_points = mutation_manager.end_or_middle_points(self.svg_path, self.mutation_points_mode)

        if len(control_points) <= 0:
            raise Exception("There is no mutation candidate")


        # select one point
        if selection_base == "random":
            # choose control point with equal weighting
            self.control_point = random.choice(control_points)
        # elif selection_base == "square-window":
        #     # choose control point based on the XAI guided weights
        #     if self.digit.attention is None:
        #         raise Exception("calculate attention map first")
        #
        #     weight_list = mutation_manager.get_attention_region_prob(self.digit.attention, control_points, SQUARE_SIZE)
        #     #print(f"weight_list: {weight_list}")
        #     if weight_list is not None:
        #         self.control_point = random.choices(population=control_points, weights=weight_list, k=1)[0]

        elif selection_base == "clustering":
            # choose control point based on the XAI guided weights
            if self.digit.attention is None:
                raise Exception("calculate attention map first")

            clustered_points, weight_list, directions\
                = mutation_manager.cluster_attention_map(self.digit.attention,
                                                         control_points,
                                                         weight_threshold=0.0001,
                                                         #previous_mask=self.digit.cluster_mask
                                                         )
            # print(f"weight_list: {weight_list}")
            if weight_list is not None:
                index = random.choices(population=range(len(control_points)), weights=weight_list, k=1)[0]
                self.control_point = control_points[index]
                self.directions = directions[index]

                # ------------------for attention reduction-------------------------
                # if self.digit.mutation_point[0] is not None:
                #     # print(control_points)
                #     # print(self.digit.mutation_point)
                #     pre_index = control_points.index(closest_2d_point(self.digit.mutation_point, control_points))
                #     # print(pre_index)
                #     # self.digit.avg_intensity_after = cluster_sum_intensity[pre_index]
                #     self.digit.rel_intensity_after = previous_percentage
                #
                # # self.digit.avg_intensity_before = cluster_sum_intensity[index]
                # self.digit.rel_intensity_before = cluster_rel_intensity[index]
                self.digit.cluster_mask = (clustered_points == index)

        # self.digit.mutation_point = self.control_point


    def mutate(self, reference=None, selection=XMUTANT_CONFIG['selection'], direction = XMUTANT_CONFIG['direction']):
        # Select mutation operator.
        # rand_mutation_probability = random.uniform(0, 1)
        # if rand_mutation_probability >= MUTOPPROB:
        #     mutation = 1
        # else:
        #     mutation = 2

        condition = True
        counter_mutations = 0  # in case of dead loop
        distance_inputs = 0
        while condition:
            counter_mutations += 1
            self.control_points_select(selection_base = selection)
            original_coordinates_str = None
            mutated_coordinates_str = None
            if self.mutation_points_mode == "end":
                # backward_centroid will reach the boundary easily, which will lead to infinite loop...
                original_coordinates_str, mutated_coordinates_str, mutated_coordinate = (
                    mutation_manager.mutate_one_point(self.control_point, direction=self.directions, mutation_direction=direction))

                if type(mutated_coordinate) is str and 0 <= float(mutated_coordinate) <= 28:
                    # valid mutation
                    condition = False
                elif type(mutated_coordinate) is tuple:
                    if 0 <= float(mutated_coordinate[0]) <= 28 and 0 <= float(mutated_coordinate[1]) <= 28:
                        condition = False

            elif self.mutation_points_mode == "mid":
                original_coordinates_str, mutated_coordinates_str, mutated_coordinate = (
                    mutation_manager.mutate_one_point(self.control_point, direction=self.directions, mutation_direction=direction))
                condition = False
            if original_coordinates_str is None or original_coordinates_str not in self.svg_path:
                raise Exception("Oops, cannot find the position of mutated point")

            mutant_vector = self.svg_path.replace(str(original_coordinates_str), str(mutated_coordinates_str))
        #     mutant_vector = mutation_manager.mutate(self.digit.xml_desc, mutation, counter_mutations/20)
            mutant_xml_desc = vectorization_tools.create_svg_xml(mutant_vector)
            rasterized_digit = rasterization_tools.rasterize_in_memory(mutant_xml_desc)

            # distance_inputs = get_distance(self.digit.purified, rasterized_digit) # not necessary
            # Not likely to be zero
            # if distance_inputs != 0:
            if reference is not None:
                distance_inputs = get_distance(reference.purified, rasterized_digit)
                if distance_inputs == 0:
                    #condition = False
                    condition = True
            # else:
            #     condition = False

            # Note: the endpoints should be within the range of [0,28],
            # while the intermediate points are not necessary.
            if counter_mutations > 10:
                raise Exception("Oops, budget exceeded, cannot find a valid mutation")

        self.digit.xml_desc = mutant_xml_desc
        self.digit.purified = rasterized_digit
        self.digit.predicted_label = None
        self.digit.confidence = None
        self.digit.correctly_classified = None

        return distance_inputs

    def generate(self):
        # Select mutation operator.
        rand_mutation_probability = random.uniform(0, 1)
        if rand_mutation_probability >= MUTOPPROB:
            mutation = 1
        else:
            mutation = 2

        condition = True
        counter_mutations = 0
        distance_inputs = 0
        while condition:
            counter_mutations += 1
            vector1, vector2 = mutation_manager.generate(
                self.digit.xml_desc,
                mutation)
            v1_xml_desc = vectorization_tools.create_svg_xml(vector1)
            rasterized_digit1 = rasterization_tools.rasterize_in_memory(v1_xml_desc)

            v2_xml_desc = vectorization_tools.create_svg_xml(vector2)
            rasterized_digit2 = rasterization_tools.rasterize_in_memory(v2_xml_desc)

            distance_inputs = get_distance(rasterized_digit1,
                                           rasterized_digit2)

            if distance_inputs != 0:
                condition = False

        first_digit = MnistMember(v1_xml_desc,
                                  self.digit.expected_label,
                                  self.seed)
        second_digit = MnistMember(v2_xml_desc,
                                   self.digit.expected_label,
                                   self.seed)
        first_digit.purified = rasterized_digit1
        second_digit.purified = rasterized_digit2
        return first_digit, second_digit, distance_inputs


